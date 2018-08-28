#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import tensorflow as tf
import datetime

#from tensorflow.python.keras.models import Model
from keras.models import Model

import Dataset, Experiment, metrics
from architectures.Residual_Network import Resnet_2x4, Resnet_2x4_poisson, Resnet_2x4_binomial

def adience_baseline_experiment(dataset):
    """Ejecuta un experimento entero (entrenamiento y test) con la configuración baseline con el dataset Adience.

    :param dataset: Dataset para realizar el experimento.
    """
    #    RUN ADIENCE BASELINE

    #Create net architecture
    baseline_resnet = Resnet_2x4()

    model = Model(inputs=baseline_resnet.inputs,
                  outputs=baseline_resnet.get_net())

    #Create experiment
    experiment = Experiment.Experiment(dataset, model)

    #Train
    experiment.train('adience_baseline', '/TFG/ordinal_unimodal_mio/src/logs/')

    #Test
    experiment.test()

def adience_baseline_experiment_sgd(dataset):
    """Ejecuta un experimento entero (entrenamiento y test) con la configuración baseline con el optimizador SGD de nesterov con el dataset Adience.

    :param dataset: Dataset para realizar el experimento.
    """
    #    RUN ADIENCE BASELINE

    #Create net architecture
    baseline_resnet = Resnet_2x4()

    model = Model(inputs=baseline_resnet.inputs,
                  outputs=baseline_resnet.get_net())

    #Create experiment
    experiment = Experiment.Experiment(dataset, model)

    arguments={'epochs': 100, 'optimizer': SGD,
               'learning_rate': 1e-2, 'momentum': 0.9,
               'loss_fn': 'categorical_crossentropy',
               'metrics': ['accuracy']}

    callbacks = ['ModelCheckpoint']

    #Train
    experiment.train('adience_baseline', '/TFG/ordinal_unimodal_mio/src/logs/', arguments=arguments, callbacks=callbacks)

    #Test
    experiment.test()

def adience_poisson_experiment(dataset, tau_mode, tau=1.):
    """Ejecuta un experimento entero (entrenamiento y test) con la configuración poisson con el dataset Adience.

    :param dataset: Dataset para realizar el experimento.
    :param tau_mode: Modo de ejecución del parámetro tau en el experimento (constante o aprender valor)
    :param tau: Valor del parámetro tau. Valor inicial si se va a aprender, o valor constante.
    """
    #    RUN ADIENCE POISSON
    assert tau_mode in ["non_learnable", "sigm_learnable"]

    #Create net architecture
    poisson_resnet = Resnet_2x4_poisson(tau_mode)

    model = Model(inputs=poisson_resnet.inputs,
                  outputs=poisson_resnet.get_net())

    #Create experiment
    experiment = Experiment.Experiment(dataset, model)

    #Train
    experiment.train('adience_poisson_t='+tau_mode, '/TFG/ordinal_unimodal_mio/src/logs/')

    #Test
    experiment.test()

def adience_binomial_experiment(dataset, tau_mode, tau=1.):
    """Ejecuta un experimento entero (entrenamiento y test) con la configuración binomial con el dataset Adience.

    :param dataset: Dataset para realizar el experimento.
    :param tau_mode: Modo de ejecución del parámetro tau en el experimento (constante o aprender valor)
    :param tau: Valor del parámetro tau. Valor inicial si se va a aprender, o valor constante.
    """
    #    RUN ADIENCE BINOMIAL
    assert tau_mode in ["non_learnable", "sigm_learnable"]

    #Create net architecture
    binomial_resnet = Resnet_2x4_binomial(tau_mode)

    model = Model(inputs=binomial_resnet.inputs,
                  outputs=binomial_resnet.get_net())

    #Create experiment
    experiment = Experiment.Experiment(dataset, model)

    #Train
    experiment.train('adience_binomial_t='+tau_mode, '/TFG/ordinal_unimodal_mio/src/logs/')

    #Test
    experiment.test()

def main():
    """Función que realiza secuencialmente todos los experimentos de los que consiste la parte experimental del trabajo.
    """
    #Load dataset
    dataset_path = "/Datasets/Adience/Adience_h5/"
    dataset = Dataset.Dataset(train_filename=dataset_path+'adience_256.h5',
                              test_filename=dataset_path+'adience_test_256.h5',
                              batch_size=128)

    #
    #    Baseline experiments
    #
    print(' # Runing Baseline experiment.')
    adience_baseline_experiment(dataset)

    print(' # Runing Baseline experiment with sgd.')
    adience_baseline_experiment_sgd(dataset)

    #
    #    Poisson experiments
    #
    print(' # Runing Poisson (t=1) experiment.')
    adience_poisson_experiment(dataset, 'non_learnable')

    print(' # Runing Poisson (t=learned) experiment.')
    adience_poisson_experiment(dataset, 'sigm_learnable')

    #
    #    Binomial experiments
    #
    print(' # Runing Binomial (t=1) experiment.')
    adience_binomial_experiment(dataset, 'non_learnable')

    print(' # Runing Binomial (t=learned) experiment.')
    adience_binomial_experiment(dataset, 'sigm_learnable')

if __name__ == '__main__':
    main()
