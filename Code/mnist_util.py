'''minst_util.py, von Reto Merz, 2018

stellt Funktionen zum Umgang mit MNIST-Daten zur Verfügung

MNIST-Datensatz: http://yann.lecun.com/exdb/mnist/
loader.py: https://pypi.python.org/pypi/python-mnist
'''


import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

def training_data(datapath):
    '''importiert MNIST Trainingsdaten aus angegebenem Verzeichnis 
        (mit Daten von http://yann.lecun.com/exdb/mnist/) unter Verwendung
        von MNIST aus loader.py von "python-mnist-0.5" 
        (https://pypi.python.org/pypi/python-mnist)'''
    try:
        from loader import MNIST
    except ModuleNotFoundError:
        raise ModuleNotFoundError('loader.py von python-mnist-0.5 ' +
            '(https://pypi.python.org/pypi/python-mnist) nicht gefunden!')

    # Daten laden
    mnist_data = MNIST(path=datapath, return_type='numpy', gz=True)
    images, labels = mnist_data.load_training()
    
    # Pixeldaten anpassen
    images = np.float64(images) / 255 # Helligkeit von 0 - 255 zu 0 - 1
    
    # Labels ins OneHot-Format umwandeln
    oh_labels = []
    for l in labels:
        z = np.zeros(10)
        z[l] = 1
        oh_labels.append(z)

    data = list(zip(images, oh_labels)) # Bilder und Labels kombinieren
    print('*Trainingsdaten geladen')
    return data

def testing_data(datapath):
    '''importiert MNIST Testdaten aus angegebenem Verzeichnis 
        (mit Daten von http://yann.lecun.com/exdb/mnist/) unter Verwendung
        von MNIST aus loader.py von "python-mnist-0.5" 
        (https://pypi.python.org/pypi/python-mnist)'''
    try:
        from loader import MNIST
    except ModuleNotFoundError:
        raise ModuleNotFoundError('loader.py von python-mnist-0.5 ' +
            '(https://pypi.python.org/pypi/python-mnist) nicht gefunden!')

    # Daten laden
    mnist_data = MNIST(path=datapath, return_type='numpy', gz=True)
    images, labels = mnist_data.load_testing()
    
    # Pixeldaten anpassen
    images = np.float64(images)
    images = images / 255 # Helligkeit von 0 - 255 zu 0 - 1

    # Labels ins OneHot-Format umwandeln
    oh_labels = []
    for l in labels:
        z = np.zeros(10)
        z[l] = 1
        oh_labels.append(z)

    data = list(zip(images, oh_labels)) # Bilder und Labels kombinieren
    print('*Testdaten geladen')
    return data

def toPKL(path):
    '''erstellt schnell abrufbare PKL-Dateien mit den MNIST-Daten, 
        path ist Verzeichnis mit MNIST-Daten von 
        http://yann.lecun.com/exdb/mnist/'''
    traindata = training_data(path)
    testdata = testing_data(path)

    with open(path + '/mnist_train.pkl', 'wb') as file:
            pickle.dump(traindata, file)

    with open(path + '/mnist_test.pkl', 'wb') as file:
            pickle.dump(testdata, file)


def draw(image, title=None):
    '''zeigt ein MNIST-Bild'''
    pixels = image.reshape((28,28)) # Vektor (1x784) zu Matrix (28x28)
    plt.imshow(pixels, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    
    if title is not None:
        plt.title(title)

    plt.show()


if __name__ == '__main__':
    '''Wenn die Datei direkt ausgeführt wird, werden MNIST-Daten im
        Verzeichnis des Programms in .pkl-Dateien umgewandelt.'''
    import os

    path = os.getcwd()
    toPKL(path)
