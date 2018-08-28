# -*- coding: utf-8 -*-

from __future__ import division

import datetime
import os

import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
                            TensorBoard, CSVLogger, ModelCheckpoint

import metrics
import helper_functions
from Data_augmentation import dataset_generator


class Experiment():
    """Clase que contiene la funcionalidad necesaria para ejecutar un experimento, es decir,
    entrenar una red deseada con un dataset elegido y obtener los resultados de test.

    :param dataset: Objeto de la clase :class:`Dataset` previamente inicializado con los nombres de ficheros, y todas las variables necesarias.
    :param net_model: Función que contiene la arquitectura de la red. Debe ser un objeto Model de keras previo a ser compilado.
    :param batch_size: Tamaño de batch.
    :param seed: Valor de la semilla aleatoria.
    """
    def __init__(self, dataset, net_model, batch_size=128, seed=1):
        self.dataset = dataset
        self.net_model = net_model

        self.batch_size = batch_size
        self.seed = seed

        self.test_metrics = {'ccr':  -1,
                             'top-2':-1,
                             'top-3':-1,
                             'qwk': float('nan')}

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

    def train(self, model_name, logs_fold, callbacks=['ReduceLROnPlateau', 'ModelCheckpoint'],
                                           arguments={'epochs':100, 'optimizer': Adam,
                                                      'learning_rate': 1e-3, 'min_lr': 1e-4,
                                                      'loss_fn': 'categorical_crossentropy',
                                                      'metrics': ['accuracy']}):
        """Realiza el entrenamiento de la red con los parámetros especificados.

        :param model_name: Nombre del model a entrenar, usado para guardar los pesos, logs, etc.
        :param logs_fold: Carpeta en la que guardar los ficheros de log.
        :param callbacks: Callbacks de keras a utilizar durante el entrenamiento.
        :param arguments: Dicionario que contiene toda la information necesaria para entrenar la red
            (épocas, optimizador, learning rate, funciones de pérdida y métricas de evaluación).
        """

        #Store variables
        self.num_epochs = arguments['epochs']
        self.optimizer_fn = arguments['optimizer']
        self.lr = arguments['learning_rate']
        self.loss_fn = arguments['loss_fn']
        self.metrics = arguments['metrics']

        self.model_name = model_name

        self.logs_fold = logs_fold

        #Create optimizer
        if arguments['optimizer'] == Adam:
            self.optimizer = self.optimizer_fn(self.lr)
        elif arguments['optimizer'] == SGD:
            self.optimizer = self.optimizer_fn(self.lr, arguments['momentum'])

        #Compile model
        self.net_model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                               metrics=self.metrics)

        #Initialize logs stuff
        self._init_logs()

        self.callbacks = []
        if 'ReduceLROnPlateau' in callbacks:
            self.min_lr = arguments['min_lr']
            self.callbacks.append(ReduceLROnPlateau(factor=0.1, verbose=1, mode='min',
                                                    patience=10, epsilon=1e-4, min_lr=self.min_lr))
        if 'ModelCheckpoint' in callbacks:
            self.callbacks.append(ModelCheckpoint(filepath=self.model_name_save, monitor='val_loss',
                                                  verbose=0, save_best_only=True,
                                                  save_weights_only=False, mode='min'))
        self.callbacks.append(CSVLogger(filename=self.log_csv))
        self.callbacks.append(TensorBoard(log_dir=self.log_tb))


        #If the data has not already been loaded, load it
        if not self.dataset.loaded:
            self.dataset.load_train()

        #
        #   Prepare variables to train
        #
        x_tr = self.dataset.train_data['input'][:]
        y_tr = helper_functions.make_onehot(self.dataset.train_data['output'][:],8)
        #y_tr = dataset.train_data['output'][:]

        x_vd = self.dataset.valid_data['input'][:]
        y_vd = helper_functions.make_onehot(self.dataset.valid_data['output'][:],8)

        # Create and init all stuff related to data augmentation
        self.train_augm = ImageDataGenerator(horizontal_flip=True)
        self.test_augm  = ImageDataGenerator(horizontal_flip=False)
        # Create dataset generator
        train_generator = dataset_generator(self.train_augm,224)(x_tr, y_tr, self.batch_size, self.dataset.num_labels, np.random.RandomState(0))
        valid_generator = dataset_generator(self.test_augm,224)( x_vd, y_vd, self.batch_size, self.dataset.num_labels)

        self.net_model.fit_generator(generator=train_generator, steps_per_epoch=self.dataset.train_batches,
                                     epochs=self.num_epochs, callbacks=self.callbacks, validation_data=valid_generator,
                                     validation_steps=self.dataset.valid_batches, verbose=2)

        #Free memory
        del x_tr
        del y_tr
        del x_vd
        del y_vd

        self.dataset._free_train()

    def test(self):
        """Ejecuta el modelo de red creado en la última iteración de entrenamiento con el conjunto de test e imprime, tanto por pantalla como a fichero diversas métricas.
        """
        #If the data has not already been loaded, load it
        if not self.dataset.loaded_test:
            self.dataset.load_test()

        x_ts = self.dataset.test_data['input'][:]
        y_ts = self.dataset.test_data['output'][:]

        test_generator = dataset_generator(self.test_augm,224)(x_ts, y_ts, self.batch_size, self.dataset.num_labels)
        predictions = self.net_model.predict_generator(test_generator, steps=self.dataset.test_batches, verbose=1)

        self.test_metrics['ccr']   = metrics.accuracy(y=y_ts, ypred=np.argmax(predictions, axis=1))
        self.test_metrics['top-2'] = metrics.topkaccuracy(y=y_ts, ypred=predictions, k=2)
        self.test_metrics['top-3'] = metrics.topkaccuracy(y=y_ts, ypred=predictions, k=3)
        self.test_metrics['qwk']   = metrics.qwkappa(y=y_ts, ypred=np.argmax(predictions, axis=1))

        self.print_file_test_metrics()
        self.print_stdout_test_metrics()


        #Free memory
        del x_ts
        del y_ts

        self.dataset._free_test()

    def test_best_model(self):
        """Ejecuta el mejor modelo de red en entrenamiento con el conjunto de test e imprime, tanto por pantalla como a fichero diversas métricas.
        """
        #Use load_model with custom layer: https://github.com/keras-team/keras/issues/4871
        from unimodal_extensions import TauLayer
        saved_model = load_model(self.model_name_save, custom_objects={'TauLayer': TauLayer})

        #If the data has not already been loaded, load it
        if not self.dataset.loaded_test:
            self.dataset.load_test()

        x_ts = self.dataset.test_data['input'][:]
        y_ts = self.dataset.test_data['output'][:]

        test_generator = dataset_generator(self.test_augm,224)(x_ts, y_ts, self.batch_size, self.dataset.num_labels)
        predictions = saved_model.predict_generator(test_generator, steps=self.dataset.test_batches, verbose=1)

        self.test_metrics['ccr']   = metrics.accuracy(y=y_ts, ypred=np.argmax(predictions, axis=1))
        self.test_metrics['top-2'] = metrics.topkaccuracy(y=y_ts, ypred=predictions, k=2)
        self.test_metrics['top-3'] = metrics.topkaccuracy(y=y_ts, ypred=predictions, k=3)
        self.test_metrics['qwk']   = metrics.qwkappa(y=y_ts, ypred=np.argmax(predictions, axis=1))

        self.print_file_test_metrics()
        self.print_stdout_test_metrics()

        #Free memory
        del x_ts
        del y_ts

        self.dataset._free_test()

    def print_stdout_test_metrics(self):
        """Imprime las métricas de test por pantalla.
        """
        #Print predictions
        print('  # Test metrics:')
        print('    * Accuracy : ' + str(self.test_metrics['ccr'] ))
        print('    * Top-2 acc: ' + str(self.test_metrics['top-2']))
        print('    * Top-3 acc: ' + str(self.test_metrics['top-3']))
        print('    * QWK      : ' + str(self.test_metrics['qwk'] ))

    def print_file_test_metrics(self):
        """Imprime las métricas de test por pantalla.
        """
        with open(self.models_test_metrics + '.csv', 'w') as f:
            f.write('Accuracy, Top-2 Accuracy, Top-3 Accuracy, QWK\n')
            f.write(str(self.test_metrics['ccr']  ) + ',')
            f.write(str(self.test_metrics['top-2']) + ',')
            f.write(str(self.test_metrics['top-3']) + ',')
            f.write(str(self.test_metrics['qwk']  ) + '\n')

    def _init_logs(self):
        """Initializa todas las variables y crea las variables relacionadas con los ficheros de logs.
        """
        self.current_time = datetime.datetime.now()
        self.current_time_str = '_'.join(str(self.current_time).split())

        self.model_name_save = 'models/' + self.model_name + '_' + self.current_time_str 

        self.config_str = self.model_name
        self.config_str += '_' + 'lr=' + str(self.lr)

        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists(self.logs_fold):
            os.makedirs(self.logs_fold)
        if not os.path.exists(self.logs_fold + 'logs_tb/'):
            os.makedirs(self.logs_fold + 'logs_tb/')
        if not os.path.exists(self.logs_fold + 'csv_logs/'):
            os.makedirs(self.logs_fold + 'csv_logs/')
        if not os.path.exists(self.logs_fold + 'models_test_metrics/'):
            os.makedirs(self.logs_fold + 'models_test_metrics/')

        self.log_tb  = self.logs_fold + 'logs_tb/'  + self.config_str +'-' + self.current_time_str
        self.log_csv = self.logs_fold + 'csv_logs/' + self.config_str +'-' + self.current_time_str + '.csv'
        self.models_test_metrics = self.logs_fold + 'models_test_metrics/' + self.model_name + '_' + self.current_time_str

