# -*- coding: utf-8 -*-

from __future__ import division

import h5py
import numpy as np
import tensorflow as tf
from math import ceil

from Data_augmentation import train_preprocess, test_preprocess
import helper_functions

class Dataset:
    """Clase que encapsula la funcionalidad necesaria de un dataset. Guarda información de estado, datos descriptivos y calcula \
     variables necesarias para su uso, como número de batchs.

    :param train_filename: Nombre del fichero que contiene los datos de entrenamiento y validación
    :param test_filename: Nombre del fichero que contiene los datos de test
    :param labels: Número de etiquetas del dataset
    :param batch_size: Tamaño de batch.
    :param num_epochs: Número de épocas para entrenar. Utilizado para repetir num_epochs veces el dataset. (-1 para repetir indefinidamente.)
    """

    def __init__(self, train_filename, test_filename, labels, batch_size=128, num_epochs=-1):
        self.train_filename = train_filename
        self.test_filename = test_filename

        self.num_labels = labels

        self.batch_size = batch_size
        self.loaded = False
        self.loaded_test = False
        self.initialized = False
        self.initialized_test = False
        self.num_epochs = num_epochs

    def _get_num_batches(self, num_images):
        """Calcula el número de batches a partir del número de imágenes y tamaño de batch.
        """
        return int(ceil(num_images/self.batch_size))

    def load_train(self):
        """Carga el conjunto de entrenamiento (y validación), calcula las dimensiones de las imágenes y el tamaño de batch.
        """
        #Load hdf5 files to variables
        self.train_file = h5py.File(self.train_filename, 'r')
        #Load train, validation and test data in dictionaries
        self.train_data = {"input": self.train_file['xt'], "output": self.train_file['yt']}
        self.valid_data = {"input": self.train_file['xv'], "output": self.train_file['yv']}

        #Get shape of images (common to all partitions)
        self.image_shape = self.valid_data['input'][0].shape
        #Get shapes for placeholders (common to all partitions)
        self.input_shape = helper_functions.get_input_shape(self.image_shape)

        #Get number of images and batches for train
        self.train_num_images = self.train_data['output'].shape[0]
        self.train_batches = self._get_num_batches(self.train_num_images)
        #Get number of images and batches for validation
        self.valid_num_images = self.valid_data['output'].shape[0]
        self.valid_batches = self._get_num_batches(self.valid_num_images)

        self.loaded = True

    def _free_train(self):
        """Libera las variables que almacenan los datos de entrenamiento y validación.
        """
        del self.train_data
        del self.valid_data

        self.loaded = False

    def load_test(self):
        """Carga el conjunto de test y calcula el tamaño de batch.
        """
        #Load hdf5 files to variables
        self.test_file =  h5py.File(self.test_filename, 'r')
        #Load test data in dictionaries
        self.test_data  = {"input": self.test_file['xtest'], "output": self.test_file['ytest']}

        #Get number of images and batches for test
        self.test_num_images = self.test_data['output'].shape[0]
        self.test_batches = self._get_num_batches(self.test_num_images)

        self.loaded_test = True

    def _free_test(self):
        """Libera las variables que almacenan los datos de test.
        """
        del self.test_data

        self.loaded_test = False

    def create_dataset_train(self):
        """Crea las instancias de *tensorflow.Dataset* con los parámetros deseados para entrenamiento y validación.
        """
        #        TRAIN
        with tf.name_scope("train_dataset_creation") as scope:
            self.features_placeholder = tf.placeholder(dtype=tf.float32, shape=self.input_shape)
            self.labels_placeholder   = tf.placeholder(dtype=tf.float32, shape=(None))

            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder,self.labels_placeholder))
            self.dataset = self.dataset.map(train_preprocess).shuffle(10000).repeat(self.num_epochs).batch(self.batch_size)

        #        VALIDATION
        with tf.name_scope("valid_dataset_creation") as scope:
            self.features_vd_placeholder=tf.placeholder(dtype=tf.float32, shape=self.input_shape)
            self.labels_vd_placeholder=tf.placeholder(dtype=tf.float32, shape=(None))

            self.dataset_vd = tf.data.Dataset.from_tensor_slices((self.features_vd_placeholder,
                                                                  self.labels_vd_placeholder))
            self.dataset_vd = self.dataset_vd.map(test_preprocess).repeat(self.num_epochs).batch(self.batch_size)

    def create_dataset_test(self):
        """Crea la instancia de *tensorflow.Dataset* con los parámetros deseados para test.
        """
        #        TEST: Reuses stuff from TRAIN
        with tf.name_scope("test_dataset_creation") as scope:
            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder,
                                                               self.labels_placeholder))
            self.dataset = self.dataset.map(test_preprocess).batch(self.batch_size)


    def initialize_train(self, session):
        """Inicializa los iteradores utilizados para los datos de entrenamiento y validación.
        
        :param session: Sesión de tensorflow para inicializar el iterador.
        """
        #Initialize iterators
        with tf.name_scope("train_iter_init") as scope:
            self.data_iterator = self.dataset.make_initializable_iterator()

            session.run(self.data_iterator.initializer,
                        feed_dict ={self.features_placeholder: self.train_data['input'],
                                    self.labels_placeholder:   self.train_data['output']})

        with tf.name_scope("valid_iter_init") as scope:
            self.data_vd_iterator = self.dataset_vd.make_initializable_iterator()

            session.run(self.data_vd_iterator.initializer,
                        feed_dict ={self.features_vd_placeholder: self.valid_data['input'],
                                    self.labels_vd_placeholder:   self.valid_data['output']})

        self.initialized = True

    def initialize_test(self, session):
        """Inicializa los iteradores utilizados para los datos de test.
        
        :param session: Sesión de tensorflow para inicializar el iterador.
        """
        with tf.name_scope("test_iter_init") as scope:
            self.data_iterator = self.dataset.make_initializable_iterator()

            session.run(self.data_iterator.initializer,
                        feed_dict ={self.features_placeholder: self.test_data['input'],
                                    self.labels_placeholder:   self.test_data['output']})

        self.initialized_test = True

#
#    Test class
#
if __name__ == '__main__':
    session = tf.InteractiveSession()
    dataset_path = "/Datasets/Adience/Adience_h5/"
    dataset = Dataset(train_filename=dataset_path+'adience_256.h5',
                        test_filename=dataset_path+'adience_test_256.h5',
                        batch_size=128, labels=-1)
    dataset.load_train()
    dataset.create_dataset_train()
    dataset.initialize_train(session)
    print("image_shape: " + str(dataset.image_shape))
    print("input_shape: " + str(dataset.input_shape))
    print("train_num_images: " + str(dataset.train_num_images))
    print("train_batches: " + str(dataset.train_batches))
    print("valid_num_images: " + str(dataset.valid_num_images))
    print("valid_batches: " + str(dataset.valid_batches))

    dataset.load_test()
    dataset.create_dataset_test()
    dataset.initialize_test(session)
    print("test_num_images: " + str(dataset.test_num_images))
    print("test_batches: " + str(dataset.test_batches))
