'''Experiment.py, von Reto Merz, 2018

Folgende Funktionen vereinfachen das Finden von geeigneten Learningrates,
das Testen von Konfigurationen und das Auswerten der generierten Daten.
'''

SAVEPATH = 'C://Users//...'
DATAPATH = 'C://Users//...'


import NeuralNetwork as N
import Functions as F

import numpy as np
import matplotlib.pyplot as plt
import csv


def lr_search(lrs, size, actfuncs, costfunc,
              n_epochs=5, batchsize=10):
    mnistnet = N.Network(size, actfuncs)
    costs = []

    for lr in lrs:
        mnistnet.reset()
        mnistnet.train(traindata, n_epochs, batchsize, costfunc, lr,
                       print_cost=False, show_cost=False)
        costs.append(mnistnet.epochcosts[-1])

    print('lrs: {}'.format(lrs))
    print('costs: {}'.format(costs))

    plt.scatter(lrs, costs)
    plt.xlabel('Lernrate')
    plt.ylabel('Kosten')
    plt.show()

def plot_results(title, *args):
    # arg = {'title': 'sample_title', 'ylabel': 'y', 'data': list_of_datalists, 'labels': ['lr=1', 'lr=2', ...]}

    fig_size = (3*len(args), 8)

    x = [(i+1) for i in range(len(args[0]['data'][0]))]
    fig = plt.figure(figsize=(fig_size[1],fig_size[0]), dpi=90)
    
    for i, arg in enumerate(args):
        ax = plt.subplot2grid(fig_size, (3*i, 0), colspan=fig_size[0], 
                              rowspan=3)
        ax.set_title(arg['title'])
        ax.set_ylabel(arg['ylabel'])
        ax.grid(True, linestyle='-')
        if i == len(args)-1:
            ax.set_xlabel('Epoche')

        if 'labels' in arg:
            if i == 0:
                ax.legend()
            for d, l in zip(arg['data'], arg['labels']):
                ax.plot(x, d, label=l)
        else:
            for d in arg['data']:
                ax.plot(x, d)
    
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def experiment(name, savepath, size, actfuncs, costfunc, lr, n_repeats=3,
               n_epochs=50, batchsize=10):
    '''trainiert mit gleicher Konfiguration mehrere Netze und speichert sie'''
    net = N.Network(size, actfuncs)
    for i in range(n_repeats):
        net.reset()
        net.train(traindata, n_epochs, batchsize, costfunc, lr, record=True,
                  test=testdata)
        N.save(net, savepath + '{}_{}.pkl'.format(name, i+1))


def auswerten(exp_name, n_nets, plot=False, write_csv=False,
              n_epochs=50, batches_per_epoch=6000, L=4):
    '''extrahiert Daten aus gespeicherten Netzen und stellt diese dar oder
        gibt sie in einer .csv-Datei aus'''
    net_names = ['{}_{}.pkl'.format(exp_name, i+1) for i in range(n_nets)]

    # Dictionaries initialisieren
    cost = {'title': 'Kosten während Training', 'ylabel': 'Kosten',
            'data': []}
    acc = {'title': 'Genauigkeit über Testdaten', 'ylabel': 'Genauigkeit [%]',
            'data': []}
    mse_test = {'title': 'MSE über Testdaten', 'ylabel': 'MSE', 'data': []}
    mse_train = {'title': 'MSE über Trainingsdaten', 'ylabel': 'MSE',
                 'data': []}
    grad_L = {'title': 'Grösse des Gradienten im letzten Layer',
              'ylabel': 'grad_size_L', 'data': []}
    grad = {key: value for key, value in [(i, []) for i in range(L-1)]}

    # Daten aus gespeicherten Netz-Klassen extrahieren
    for name in net_names:
        net = N.load(SAVEPATH + name)

        cost['data'].append(net.epochcosts)
        acc['data'].append(net.accuracys)
        mse_test['data'].append(net.mse_test)
        mse_train['data'].append(net.mse_train)

        gradient_sizes = np.array(net.gradient_sizes) # = [[gradl_0, gradL_0], [gradl_1, gradL_1], ...]
        gradient_sizes = np.transpose(gradient_sizes) # = [[gradl_0, gradl_1, ...], [gradL_0, gradL_1, ...]]

        n_batches = n_epochs * batches_per_epoch

        for l, gradient_size in enumerate(gradient_sizes):
            avg_grads_per_epoch = [np.mean(gradient_size[j:j+batches_per_epoch]) 
                               for j in range(0, n_batches, batches_per_epoch)]
            grad[l].append(avg_grads_per_epoch)

        grad_L['data'].append(avg_grads_per_epoch)

    # Daten in einer .csv-Datei speichern
    if write_csv:
        n_cols = 5 + len(grad[0])

        with open('{}.csv'.format(exp_name), 'w') as file:
            epoch_col = [exp_name, 'epoch'] 
            for i in range(n_nets):
                epoch_col += ['-{}-'.format(i+1)]
                epoch_col += [j+1 for j in range(n_epochs)]

            cost_col = [exp_name, 'cost']
            for i in range(n_nets):
                cost_col += ['-{}-'.format(i+1)]
                cost_col += cost['data'][i]

            acc_col = [exp_name, 'acc']
            for i in range(n_nets):
                acc_col += ['-{}-'.format(i+1)]
                acc_col += acc['data'][i]

            mse_test_col = [exp_name, 'mse_test']
            for i in range(n_nets):
                mse_test_col += ['-{}-'.format(i+1)]
                mse_test_col += mse_test['data'][i]

            mse_train_col = [exp_name, 'mse_train']
            for i in range(n_nets):
                mse_train_col += ['-{}-'.format(i+1)]
                mse_train_col += mse_train['data'][i]

            cols = [epoch_col, cost_col, acc_col, mse_test_col, mse_train_col]

            for l in range(L-1):
                grad_col = [exp_name, 'grad_{}'.format(l)]
                for i in range(n_nets):
                    grad_col += ['-{}-'.format(i+1)]
                    grad_col += grad[l][i]
                cols.append(grad_col)

            rows = np.transpose(cols)
            csv.writer(file, delimiter=';').writerows(rows)

    if plot:
        plot_results(exp_name, cost, acc, mse_test, mse_train, grad_L)

    return cost, acc, mse_test, mse_train, grad_L, grad




if __name__=='__main__':

	traindata = N.load(DATAPATH + 'mnist_traindata.pkl')
	testdata = N.load(DATAPATH + 'mnist_testdata.pkl')
	print('-Daten geladen-')


    conf1 = [784, 50, 10]
    conf2 = [784, 50, 50, 10]

    # ERSTER VERSUCH

    '''Lernraten (conf1):
    Sigmoid / MSE:          3

    Learning Rates (conf2):
    Sigmoid / MSE:          3
    Sigmoid / CE:           0.2
    ReLU / MSE:             0.08
    lReLU / MSE:            0.08
    Sigmoid / Softmax:      0.2
    lReLU, Softmax / CE:    0.05
    lReLU, Sigmoid / CE:    0.07
    '''

    experiment('sig_mse_c1', SAVEPATH, conf1, 
               [F.Sigmoid, F.Sigmoid], F.MSE, 3,
               n_epochs=50, n_repeats=3)

    experiment('sig_mse_c2', SAVEPATH, conf2,
               [F.Sigmoid, F.Sigmoid, F.Sigmoid], F.MSE, 3,
               n_epochs=50, n_repeats=3)
    experiment('sig_ce_c2', SAVEPATH, conf2,
               [F.Sigmoid, F.Sigmoid, F.Sigmoid], F.CrossEntropy, 0.2,
               n_epochs=50, n_repeats=3)
    experiment('relu_mse_c2', SAVEPATH, conf2,
               [F.ReLU, F.ReLU, F.ReLU], F.MSE, 0.08,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_mse_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.lReLU], F.MSE, 0.08,
               n_epochs=50, n_repeats=3)
    experiment('sig_smax_ce_c2', SAVEPATH, conf2,
               [F.Sigmoid, F.Sigmoid, F.Softmax], F.CrossEntropy, 0.2,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_smax_ce_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.Softmax], F.CrossEntropy, 0.05,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_sig_ce_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.Sigmoid], F.CrossEntropy, 0.07,
               n_epochs=50, n_repeats=3)

    auswerten('sig_mse_c1', 3, plot=True, write_csv=True, L=3)
    versuche = ['sig_mse_c2', 'sig_ce_c2', 'lrelu_mse_c2', 'sig_smax_ce_c2',
                'lrelu_smax_ce_c2', 'lrelu_sig_ce_c2']
    for name in versuche:
        auswerten(name, 3, plot=True, write_csv=True, L=4)


    # ZWEITER VERSUCH

    '''Learning Rates (conf1):
    Sigmoid / MSE:          2

    Learning Rates (conf2):
    Sigmoid / MSE:          2
    ReLU / MSE:             0.01
    lReLU / MSE:            0.05
    lReLU, Softmax / CE:    0.03
    lReLU, Sigmoid / CE:    0.03
    '''

    experiment('sig_mse_c1', SAVEPATH, conf1, 
               [F.Sigmoid, F.Sigmoid], F.MSE, 2,
               n_epochs=50, n_repeats=3)

    experiment('sig_mse_c2', SAVEPATH, conf2,
               [F.Sigmoid, F.Sigmoid, F.Sigmoid], F.MSE, 2,
               n_epochs=50, n_repeats=3)
    experiment('relu_mse_c2', SAVEPATH, conf2,
               [F.ReLU, F.ReLU, F.ReLU], F.MSE, 0.01,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_mse_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.lReLU], F.MSE, 0.05,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_smax_ce_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.Softmax], F.CrossEntropy, 0.03,
               n_epochs=50, n_repeats=3)
    experiment('lrelu_sig_ce_c2', SAVEPATH, conf2,
               [F.lReLU, F.lReLU, F.Sigmoid], F.CrossEntropy, 0.03,
               n_epochs=50, n_repeats=3)

    auswerten('sig_mse_c1', 3, plot=True, write_csv=True, L=3)
    versuche = ['sig_mse_c2', 'relu_mse_c2' 'lrelu_mse_c2',
                'lrelu_smax_ce_c2', 'lrelu_sig_ce_c2']
    for name in versuche:
        auswerten(name, 3, plot=True, write_csv=True, L=4)
