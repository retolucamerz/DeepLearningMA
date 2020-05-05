'''NeuralNetwork.py, von Reto Merz, 2018

Mit diesem Modul lassen sich beliebige feedforward neuronale Netze mit
beliebiegen Aktivierungs- und Kostenfunktionen erstellen und trainieren.

size: Liste mit Anzahl Neuronen im jeweiligen Layer
    (Bsp.: [784, 50, 10] bedeutet 784 Inputneuronen, ein Hidden Layer mit
     50 Neuronen und ein Ausgabelayer mit 10 Neuronen)

actfuncs: Liste mit Aktivierungsfunktionen (Format wie bei DLFunctions)
    (Bsp.: [F.Sigmoid, F.Sigmoid])

dataset: Datensatz mit Features und OneHot-Label in Liste
    (Bsp.: [[np.array([0.2, 0.8]), np.array([0, 1, 0])], ...])


basierend auf NumPy und matplotlib
'''

import numpy as np  # for fast matrix operations
import time  # to time


# activation functions
class Sigmoid:
    @staticmethod
    def A(z):
        return 1/(1+np.exp(-z))

    @staticmethod
    def dA_dz(z):
        a = Sigmoid.A(z)
        return a*(1-a)


class ReLU:
    @staticmethod
    def A(z):
        return np.maximum(z, 0)

    @staticmethod
    def dA_dz(z):
        return np.where(z > 0, 1, 0)


class lReLU:
    @staticmethod
    def A(z):
        return np.where(z > 0, z, z * 0.01)

    @staticmethod
    def dA_dz(z):
        return np.where(z > 0, 1, 0.01)


# cost functions
class MSE:
    @staticmethod
    def C(y, y_hat):
        return 0.5 * np.dot(y-y_hat, y-y_hat)

    @staticmethod
    def dC_da(y, y_hat):
        return y - y_hat


class CrossEntropy:
    @staticmethod
    def C(y, y_hat):
        c = -(y_hat*np.log(y)+(1-y_hat)*np.log(1-y))
        return c.sum()

    @staticmethod
    def dC_da(y, y_hat):
        return (1-y_hat)/(1-y) - y_hat/y


# neural network class definition
class Network:
    def __init__(self, size, actfuncs, rdm_state=None):
        '''initializes network and creates all lists for performance evaluation'''
        self.size = size  # list with number of neurons per layer
        self.actfuncs = actfuncs  # list with acitvation functions (per layer)

        if len(size) != len(actfuncs)+1:
            raise TypeError('len(size)!=len(actfuncs)+1')

        self.L = len(self.size)  # number of layers (including inputlayer)

        # set and save random state
        if rdm_state is not None:
            np.random.set_state(rdm_state)
        else:
            np.random.seed()
        self.rdm_state = np.random.get_state()

        # initialize weight and biases
        sigmas = [1 / np.sqrt(i) for i in self.size]
        self.weights = [np.random.normal(0, sigmas[i],
                                         (self.size[i], self.size[i+1]))
                        for i in range(self.L-1)]
        self.biases = [np.random.normal(0, 1, self.size[i+1])
                       for i in range(self.L-1)]

        # lists to save weighted sums and activations in all layers
        #   of the last prediction
        self.activations = [None for i in range(self.L)]
        self.w_sums = [None for i in range(self.L)]

        # lists to track performance during training per epoch
        self.epochcosts = []  # cost as calculated by costfunc
        self.accuracys = []  # classification accuracy over test dataset
        self.mse_test = []  # MSE cost over test dataset
        self.mse_train = []  # MSE cost over training dataset
        self.gradient_sizes = []  # average size of gradient
        self.epochs_trained = 0  # number of completed epochs

    def predict(self, features):
        '''calculates prediction and saves intermediate results'''
        self.activations[0] = features

        # feedforward
        for l in range(self.L-1):
            self.w_sums[l+1] = (np.matmul(self.activations[l],
                                          self.weights[l])
                                + self.biases[l])
            self.activations[l+1] = self.actfuncs[l].A(self.w_sums[l+1])

        return self.activations[-1]

    def clean_activations(self):
        '''sigmoid activations can actually be 0 or 1, when they are more accurate than float.'''
        self.activations[-1][self.activations[-1] == 0] = 0.000000000001
        self.activations[-1][self.activations[-1] == 1] = 0.999999999999

    @staticmethod
    def create_mini_batches(dataset, mini_batch_size):
        '''shuffles data set and devides dataset into mini-batches'''
        np.random.shuffle(dataset)
        mini_batches = [dataset[i:i+mini_batch_size]
                        for i in range(0, len(dataset), mini_batch_size)]
        return mini_batches

    def update_params(self, mini_batch, costfunc, learningrate):
        '''calculates the gradient of the mini batch and applies it to the weights and biases (SGD)'''
        self.errors = [None for i in range(self.L)]  # Error einer Datenpunktes
        self.gradient_w = [np.zeros(i.shape) for i in self.weights]
        self.gradient_b = [np.zeros(i.shape) for i in self.biases]
        self.mini_batchcost = 0

        # calculate the gradient
        for point in mini_batch:
            self.predict(point[0])
            self.clean_activations()
            self.mini_batchcost += costfunc.C(self.activations[-1], point[1])

            # error in the last layer
            self.errors[-1] = (costfunc.dC_da(self.activations[-1], point[1])
                               * self.actfuncs[-1].dA_dz(self.w_sums[-1]))

            # error in all other layers
            for i in reversed(range(2, self.L)):
                last_error = self.errors[i]
                transposed_w = np.transpose(self.weights[i-1])
                self.errors[i-1] = (self.actfuncs[i-2].dA_dz(self.w_sums[i-1])
                                    * np.matmul(last_error, transposed_w))

            # use errors to calculate gradient
            for i in range(self.L-1):  # 0, 1, ..., L-2
                a_transposed = self.activations[i].reshape((self.size[i], 1))
                error = self.errors[i+1].reshape((1, self.size[i+1]))
                self.gradient_w[i] += np.matmul(a_transposed, error)
                self.gradient_b[i] += self.errors[i+1]

        # apply gradient
        for i in range(self.L-1):
            self.weights[i] -= learningrate/len(mini_batch)*self.gradient_w[i]
            self.biases[i] -= learningrate/len(mini_batch)*self.gradient_b[i]

    @staticmethod
    def correct_classification(prediction, label):
        '''checks if prediction is correct, i.e. if it has the highest value
            where label (usually a one-hot vector) has it too'''
        if np.argmax(prediction) == np.argmax(label):
            return True
        else:
            return False

    def classification_accuracy(self, dataset):
        '''calculates the classification accuracy on dataset
            (number of correctly classified / total number)'''
        n_correct = sum(Network.correct_classification(self.predict(d[0]), d[1])
                        for d in dataset)

        accuracy = n_correct / len(dataset)
        return accuracy

    def cost_over_dataset(self, dataset, costfunc):
        '''calculates the cost of network predictions over dataset'''
        cost = 0
        for d in dataset:
            p = self.predict(d[0])
            cost += costfunc.C(p, d[1])
        cost = cost / len(dataset) / self.size[-1]
        return cost

    def train(self, dataset, n_epochs, mini_batch_size, costfunc, learningrate,
              test=None, record_analytics=False, print_cost_every=1, print_acc=False):
        '''trains the network in epochs and batches'''

        # Textausgabe in Konsole
        print('''
---Training started---
{} epochs, {} training samples: , mini-batch size: {}, learning rate: {}
'''.format(n_epochs, len(dataset), mini_batch_size, learningrate))

        start_time = time.time()  # record start time

        # train the network
        for epoch in range(n_epochs):
            epochcost = 0
            mini_batches = Network.create_mini_batches(
                dataset, mini_batch_size)

            for mini_batch in mini_batches:
                self.update_params(mini_batch, costfunc, learningrate)
                epochcost += self.mini_batchcost
                if record_analytics:
                    # calculate and save how much parameters have changed in this iteration
                    self.gradient_sizes.append([np.mean(np.absolute(i))
                                                for i in self.gradient_w])

            epochcost = epochcost / len(dataset) / self.size[-1]
            self.epochcosts.append(epochcost)
            self.epochs_trained += 1

            if print_cost_every and (epoch + 1) % print_cost_every == 0:
                print('Epoch {} of {} finished. Cost: {:.6f}'.format(
                    epoch+1, n_epochs, epochcost))

            if test is not None:
                # calculate MSE cost over test dataset
                mse = self.cost_over_dataset(test, MSE)
                self.mse_test.append(mse)

                # calculate accuracy over test dataset
                accuracy = 100 * self.classification_accuracy(test)
                self.accuracys.append(accuracy)
                if print_acc:
                    print('Accuracy: {:.3f} %'.format(accuracy))

            if record_analytics:
                # calculate and save MSE over training data
                #  (to compare cost when different cost functions have been used)
                mse = self.cost_over_dataset(dataset, MSE)
                self.mse_train.append(mse)

        # letzte Textausgabe
        if print_cost_every and not n_epochs % print_cost_every == 0:
            print('last cost: {:.6f}'.format(self.epochcosts[-1]))
        print('time: {:.2f} s'.format(time.time() - start_time))
        print('---Training finished---')
        print('')

    def reset(self, rdm_state=None):
        '''resets training progress'''
        # set and save random state
        if rdm_state is not None:
            np.random.set_state(rdm_state)
        else:
            np.random.seed()
        self.rdm_state = np.random.get_state()

        # initialize weight and biases
        sigmas = [1 / np.sqrt(i) for i in self.size]
        self.weights = [np.random.normal(0, sigmas[i],
                                         (self.size[i], self.size[i+1]))
                        for i in range(self.L-1)]
        self.biases = [np.random.normal(0, 1, self.size[i+1])
                       for i in range(self.L-1)]

        # lists to save weighted sums and activations in all layers
        #   of the last prediction
        self.activations = [None for i in range(self.L)]
        self.w_sums = [None for i in range(self.L)]

        # lists to track performance during training per epoch
        self.epochcosts = []  # cost as calculated by costfunc
        self.accuracys = []  # classification accuracy over test dataset
        self.mse_test = []  # MSE cost over test dataset
        self.mse_train = []  # MSE cost over training dataset
        self.gradient_sizes = []  # average size of gradient
        self.epochs_trained = 0  # number of completed epochs

    def n_vars(self):
        '''calculates the number of trainable parameters in the network'''
        n = self.size[0]  # number of activations in input layer
        for i in range(1, len(self.size)):
            n += self.size[i]*self.size[i-1]    # weights
            n += self.size[i]                   # biases
            # n += self.size[i]                   # weighted sums
            # n += self.size[i]                   # activations
        return n


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    def generate_sample_data(n, spread=0.3):
        '''creates sample training data: points on plane'''

        replication_factor = int((n-20)/20)

        # list of starting points
        d_0 = [(0.26, 0.85), (0.54, 0.88), (0.81, 0.69), (0.88, 0.46), (0.14, 0.63),
               (0.15, 0.42), (0.26, 0.22), (0.62, 0.17), (0.82, 0.34), (0.44, 0.19)]
        # artificially increase number of points by adding close points to existing ones
        for i in range(replication_factor):
            for j in range(10):
                d_0.append((d_0[j][0] + spread*(np.random.random()-0.5),
                            d_0[j][1] + spread*(np.random.random()-0.5)))

        # same with other starting points
        d_1 = [(0.4, 0.63), (0.56, 0.62), (0.57, 0.53), (0.49, 0.58), (0.39, 0.53),
               (0.48, 0.49), (0.49, 0.54), (0.57, 0.49), (0.64, 0.55), (0.49, 0.45)]
        for i in range(replication_factor):
            for j in range(10):
                d_1.append((d_1[j][0] + spread*(np.random.random()-0.5),
                            d_1[j][1] + spread*(np.random.random()-0.5)))

        # bring data in the right shape and shuffle
        data_0 = [[np.clip(np.array(d_0[i], dtype=np.float64), 0.01, 0.99), 0]
                  for i in range(len(d_0))]
        data_1 = [[np.clip(np.array(d_1[i], dtype=np.float64), 0.01, 0.99), 1]
                  for i in range(len(d_1))]
        data = data_0 + data_1
        random.shuffle(data)

        return data

    def dataToXY(data):
        '''converts training data to lists of coordinates of points;
            used for plotting training data'''
        x1 = []
        x2 = []
        y1 = []
        y2 = []

        for d in data:
            if d[1] == 0:
                x1.append(d[0][0])
                y1.append(d[0][1])

            if d[1] == 1:
                x2.append(d[0][0])
                y2.append(d[0][1])

        return x1, x2, y1, y2

    def plot_training_performance(net):
        x = list(range(1, net.epochs_trained+1))

        fig, ax1 = plt.subplots()

        plt.title('Evolution of Network Performance')

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost', color=color)
        ax1.plot(x, net.epochcosts, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0)

        if len(net.accuracys) == net.epochs_trained:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            # we already handled the x-label with ax1
            ax2.set_ylabel('Accuracy [%]', color=color)
            ax2.plot(x, net.accuracys, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(top=100)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def plot_network_predictions(net, data, resolution=200):
        xx, yy = np.meshgrid(
            np.linspace(0, 1, resolution+1),
            np.linspace(0, 1, resolution+1))

        zz = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = net.predict([xx[i, j], yy[i, j]])

        plt.pcolor(xx, yy, zz)
        plt.colorbar()

        x1, x2, y1, y2 = dataToXY(data)
        plt.scatter(x1, y1, label='Training Data 0', color='r', marker='x')
        plt.scatter(x2, y2, label='Training Data 1', color='g', marker='x')
        plt.legend()

        plt.title('Predictions of Neural Network')
        plt.xticks([])
        plt.yticks([])

        plt.show()

    print()

    data = generate_sample_data(300, spread=0.2)

    net = Network([2, 40, 1], [Sigmoid, Sigmoid])

    print(f'The network has {net.n_vars()} trainable parameters.')

    net.train(data, n_epochs=500, costfunc=CrossEntropy, learningrate=1.5,
              mini_batch_size=100, print_cost_every=50, test=data)

    plot_training_performance(net)
    plot_network_predictions(net, data)
