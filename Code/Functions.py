'''Functions.py, von Reto Merz, 2018

(vektorisierte) Aktivierungs- und Kostenfunktionen für Deep Learning

Funktionen haben folgendes Format:

- Aktivierungsfunktionen
class Actfunc(object):
    
    @staticmethod
    def A(z):
        # Aktivierung aus Vektor z berechnen
        ...
        return activation

    @staticmethod
    def dA_dz(z):
        # Ableitung der Aktivierung des Vektors z berechnen
        ...
        return dA_dz

- Kostenfunktionen
class Costfunc(object):

    @staticmethod
    def C(a, y):
        # berechnet Kosten einer Vorhersage
        ...
        return cost

    @staticmethod
    def dC_da(a, y):
        # berechnet den dC_da-Vektor
        ...
        return dC_da
'''


import numpy as np

# AKTIVIERUNGSFUNKTIONEN

class Unit:

    @staticmethod
    def A(z):
        return z

    @staticmethod
    def dA_dz(z):
        return np.ones(z.shape)


class Step:

    @staticmethod
    def A(z):
        return np.where(z >= 0, 1, 0)

    @staticmethod
    def dA_dz(z):
        return np.zeros(z.shape)


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


class Softmax:

    @staticmethod
    def A(z):
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    @staticmethod
    def dA_dz(z):
        a = Softmax.A(z)
        return a*(1-a)


# KOSTENFUNKTIONEN

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

