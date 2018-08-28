# -*- coding: utf-8 -*-

from __future__ import division

import warnings
import numpy as np

from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics import accuracy_score

# This really need to be imported if it is merged with sklearn
import scipy.stats

def qwkappa(y, ypred):
    """Calcula el Quadratic Wweighted Kappa para la clasificación realizada por la red.
    
    :param y: Vector de n elementos con los valores de clase reales de cada patrón.
    :param ypred: Matriz de nxk con las probabilidades de pertenencia o clases predichas de cada patrón a cada clase.
    :return: Valor de QWK para la clasificación realizada.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cm = confusion_matrix(y, ypred)
        n_class = cm.shape[0]
        costes=np.reshape(np.tile(range(n_class),n_class),(n_class,n_class))
        costes = (costes - costes.T)**2
        f = 1 - costes

        n = cm.sum()
        x = cm/n

        r = x.sum(axis=1) # Row sum
        s = x.sum(axis=0) # Col sum
        Ex = r.reshape(-1, 1) * s
        po = (x * f).sum()
        pe = (Ex * f).sum()
        return (po - pe) / (1 - pe)


def accuracy(y, ypred):
    """Computa el acierto del clasificador.
    
    :param y: Vector de n elementos con los valores de clase reales de cada patrón.
    :param ypred: Matriz de nxk con las probabilidades de pertenencia o array con las clases predichas de cada patrón a cada clase.
    :return: Porcentaje de acierto calculado
    """
    if len(ypred.shape) == 1:
        return accuracy_score(y, ypred)
    else:
        return topkaccuracy(y, ypred)

def topkaccuracy(y, ypred, k=1, prob=False):
    """Computa el acierto del clasificador tomando como aciertos los k valores clasificados con mayor probabilidad, es decir, las k clases que tengan mayor probabilidad.
    
    :param y: Vector de n elementos con los valores de clase reales de cada patrón.
    :param ypred: Matriz de nxk con las probabilidades de pertenencia de cada patrón a cada clase.
    :param k: Numero de las primeras k clases a comprobar.
    :return: Porcentaje de acierto calculado
    """

    if k > len(y):
        print(" Error: K mayor al número de clases!!")
        raise IndexError()
    elif k == 1 and len(ypred.shape) == 1:
        return accuracy_score(y, ypred)

    assert len(y) == ypred.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if k == 1:
            top_k_pred = np.argsort(ypred)[:,-1:]
        else:
            top_k_pred = np.argsort(ypred)[:,-k:]
        y_true = y.reshape(y.shape[0],1)

        correct = (y_true == top_k_pred).sum()
        total = len(y)

        return correct / total
