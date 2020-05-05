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


import pickle  #  zum Speichern/Laden von Ojekten
import time  # zum Messen von Zeiten

import numpy as np  # für Matrixoperationen
import matplotlib.pyplot as plt  # zum Erstellen von Grafiken

import Functions as F


# HELFERFUNKTIONEN

def simple_plot(y, title, xlabel, ylabel, start_x=1):
    '''erstellt eine Grafik mit den angegebenen Parametern,
        der Abstand in y-Richtung beträgt 1'''
    x = [i+start_x for i in range(len(y))]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='-')
    plt.show()


class Network:
    '''Instanzen dieser Klasse sind neuronale Netze'''

    def __init__(self, size, actfuncs, rdm_state=None):
        '''erstellt alle benötigten Listen und initialisiert
            die Gewichte/Neigungen'''
        self.size = size  #  Liste mit Anzahl Neuronen pro Layer
        self.actfuncs = actfuncs  # Liste mit Aktivierungsfunktionen pro Layer

        if len(size) != len(actfuncs)+1:
            raise TypeError('len(size)!=len(actfuncs)+1')

        self.L = len(self.size)  #  Anzahl Layer (Inputlayer inbegriffen)

        # Zufallszustand speichern
        if rdm_state is not None:
            np.random.set_state(rdm_state)
        else:
            np.random.seed()
        self.rdm_state = np.random.get_state()

        # Gewichte/Neigungen initialisieren
        sigmas = [1 / np.sqrt(i) for i in self.size]
        self.weights = [np.random.normal(0, sigmas[i],
                                         (self.size[i], self.size[i+1]))
                        for i in range(self.L-1)]
        self.biases = [np.random.normal(0, 1, self.size[i+1])
                       for i in range(self.L-1)]

        # Listen zum speichern von gewichteten Summen und Aktivierungen
        #   aller Layer der zuletzt durchgeführten Vorhersage
        self.activations = [None for i in range(self.L)]
        self.w_sums = [None for i in range(self.L)]

        # Listen zur Datenerfassung während dem Training
        self.epochcosts = []  # Kosten während Training pro Epoch
        self.accuracys = []  # Klassifikationsgenauigkeit über Testdaten
        self.mse_test = []  # MSE-Kosten über den Testdaten (nach jedem Epoch)
        self.mse_train = []  # MSE-Kosten über den Trainingsdaten
        self.gradient_sizes = []  # durchschnittliche Grösse des  Gradienten
        self.epochs_trained = 0  # Anzahl trainierte Epochs

    def predict(self, features):
        '''berechnet Vorhersage und speichert Zwischenresultate in Listen'''
        self.activations[0] = features

        # feedforward
        for l in range(self.L-1):
            self.w_sums[l+1] = (np.matmul(self.activations[l],
                                                self.weights[l])
                                      + self.biases[l])
            self.activations[l+1] = self.actfuncs[l].A(self.w_sums[l+1])

        return self.activations[-1]

    def clean_activations(self):
        '''ersetzt Nullen und Einsen aus den Aktivierungen des letzten
            Layers, da es ansonsten zum Teilen durch Null kommen kann'''
        self.activations[-1][self.activations[-1] == 0] = 0.000000000001
        self.activations[-1][self.activations[-1] == 1] = 0.999999999999

    @staticmethod
    def create_mini_batches(dataset, mini_batchsize):
        '''mischt Datenset zufällig und teilt Daten in Mini-Batches ein'''
        np.random.shuffle(dataset)
        mini_batches = [dataset[i:i+mini_batchsize]
                        for i in range(0, len(dataset), mini_batchsize)]
        return mini_batches

    def update_params(self, mini_batch, costfunc, learningrate):
        '''berechnet den Gradienten eines Mini-Batches mit Backpropagation
            und wendet ihn auf die Parameter an (SGD)'''
        self.errors = [None for i in range(self.L)]  # Error einer Datenpunktes
        self.gradient_w = [np.zeros(i.shape) for i in self.weights]
        self.gradient_b = [np.zeros(i.shape) for i in self.biases]
        self.mini_batchcost = 0

        # Gradient des Mini-Batches bestimmen
        for point in mini_batch:
            self.predict(point[0])
            self.clean_activations()
            self.mini_batchcost += costfunc.C(self.activations[-1], point[1])

            # Fehler im letzten Layer berechnen
            self.errors[-1] = (costfunc.dC_da(self.activations[-1], point[1])
                               * self.actfuncs[-1].dA_dz(self.w_sums[-1]))

            # Fehler in den vorderen Layern berechnen
            for i in reversed(range(2, self.L)):
                last_error = self.errors[i]
                transposed_w = np.transpose(self.weights[i-1])
                self.errors[i-1] = (self.actfuncs[i-2].dA_dz(self.w_sums[i-1])
                                    * np.matmul(last_error, transposed_w))

            # Gradient aus Fehler berechnen
            for i in range(self.L-1):  # 0, 1, ..., L-2
                a_transposed = self.activations[i].reshape((self.size[i], 1))
                error = self.errors[i+1].reshape((1, self.size[i+1]))
                self.gradient_w[i] += np.matmul(a_transposed, error)
                self.gradient_b[i] += self.errors[i+1]

        # Gradient auf Parameter anwenden
        for i in range(self.L-1):
            self.weights[i] -= learningrate/len(mini_batch)*self.gradient_w[i]
            self.biases[i] -= learningrate/len(mini_batch)*self.gradient_b[i]

    @staticmethod
    def correct_classification(prediction, label):
        '''gibt True zurück, wenn die Vorhersage stimmt
            (grösster Wert an gleicher Stelle), sonst False'''
        if np.argmax(prediction) == np.argmax(label):
            return True
        else:
            return False

    def classification_accuracy(self, dataset):
        '''berechnet die Klassifikationsgenauigkeit
            (Anzahl richtig Erkannte / Gesamtzahl) bei einem Test-Datenset'''
        n_correct = 0
        for d in dataset:
            p = self.predict(d[0])
            if Network.correct_classification(p, d[1]):
                n_correct += 1

        accuracy = n_correct / len(dataset)
        return accuracy

    def cost_over_dataset(self, dataset, costfunc):
        '''berechnet Kosten über Datenset'''
        cost = 0
        for d in dataset:
            p = self.predict(d[0])
            cost += costfunc.C(p, d[1])
        cost = cost / len(dataset) / self.size[-1]
        return cost

    def train(self, dataset, n_epochs, mini_batchsize, costfunc, learningrate,
              test=None, record=False, print_cost=True,
              show_cost=False, show_acc=False):
        '''trainiert das Netzwerk in Epochen und Batches'''

        # Textausgabe in Konsole
        print('''
---Training gestartet---
{} Epochen, Datensatzbeispiele: {}, Mini-Batch-Grösse: {}, Lernrate: {}
'''.format(n_epochs, len(dataset), mini_batchsize, learningrate))

        start_time = time.time()  # Startzeit erfassen

        # Trainingsprozess
        for epoch in range(n_epochs):
            epochcost = 0
            mini_batches = Network.create_mini_batches(dataset, mini_batchsize)

            for mini_batch in mini_batches:
                self.update_params(mini_batch, costfunc, learningrate)
                epochcost += self.mini_batchcost
                if record:
                    self.gradient_sizes.append([np.mean(np.absolute(i))
                                                for i in self.gradient_w])

            epochcost = epochcost / len(dataset) / self.size[-1]
            self.epochcosts.append(epochcost)
            self.epochs_trained += 1

            if print_cost:
                # Zwischenbericht ausgeben
                print('Epoche {} von {} fertig. Kosten: {:.6f}'.format(
                        epoch+1, n_epochs, epochcost))

            if test is not None:
                # MSE über Testdaten berechnen
                mse = self.cost_over_dataset(test, F.MSE)
                self.mse_test.append(mse)

                # Genauigkeit über Daten aus "test" bestimmen
                accuracy = 100 * self.classification_accuracy(test)
                self.accuracys.append(accuracy)
                if show_acc:
                    print('Genauigkeit: {:.3f} %'.format(accuracy))

            if record:
                # MSE über Trainingsdaten berechnen
                mse = self.cost_over_dataset(dataset, F.MSE)
                self.mse_train.append(mse)

        # letzte Textausgabe
        if not print_cost:
            print('letzte Kosten: {:.6f}'.format(self.epochcosts[-1]))
        print('Zeit: {:.2f} s'.format(time.time() - start_time))
        print('---Training abgeschlossen---')
        print('')

        if show_cost:
            simple_plot(self.epochcosts, 'Kosten während dem Training',
                        'Epoche', 'Kosten')

        if show_acc:
            simple_plot(self.accuracys, 'Entwicklung der Genauigkeit',
                        'Epoche', 'Genauigkeit [%]')

    def reset(self, rdm_state=None):
        '''setzt die wichtigsten Attribute eines Netzes zurück'''

        # Zufallszustand speichern
        if rdm_state is not None:
            np.random.set_state(rdm_state)
        else:
            np.random.seed()
        self.rdm_state = np.random.get_state()

        # Gewichte/Neigungen initialisieren
        sigmas = [1 / np.sqrt(i) for i in self.size]
        self.weights = [np.random.normal(0, sigmas[i],
                                         (self.size[i], self.size[i+1]))
                        for i in range(self.L-1)]
        self.biases = [np.random.normal(0, 1, self.size[i+1])
                       for i in range(self.L-1)]

        # Listen zum speichern von gewichteten Summen und Aktivierungen
        #   aller Layer der letzten Vorhersage
        self.activations = [None for i in range(self.L)]
        self.w_sums = [None for i in range(self.L)]

        # Listen zur Datenerfassung während dem Training
        self.epochcosts = []  # Kosten während Training pro Epoch
        self.accuracys = []  # Klassifikationsgenauigkeit über Testdaten
        self.mse_test = []  # MSE-Kosten über den Testdaten
        self.mse_train = []  # MSE-Kosten über den Trainingsdaten
        self.gradient_sizes = []  # durchschnittliche Grösse des Gradienten
        self.epochs_trained = 0  # Anzahl trainierte Epochs


    def n_vars(self):
        '''berechnet die Anzhal Parameter eines neuronalen Netzes'''
        n = size[0]  # Aktivierungen im Inputlayer
        for i in range(1, len(self.size)):
            n += self.size[i]*self.size[i-1]    # Gewichte
            n += self.size[i]                   # Neigungen
            n += self.size[i]                   # Gewichte Summen
            n += self.size[i]                   # Aktivierungen
        return n


# SPEICHERN / LADEN

def save(obj, path):
    '''speichert beliebige Objekte, also auch Instanzen der Netzwerkklasse'''
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load(path):
    '''ladet Objekte von einer Datei'''
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj
