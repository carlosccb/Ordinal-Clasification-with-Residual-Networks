# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def make_onehot(labels, num_labels):
    """Función que codifica un array de etiquetas a forma one-hot.
    
    :param labels: Array que contiene las etiquetas a transformar.
    :param num_labels: Número de clases del problema tratado.
    :return: Array con la codificación one-hot del array de entrada.
    """
    labels = labels.ravel()
    one_hot = np.eye(num_labels)[labels]

    return one_hot

def get_input_shape(img_shape):
    """Obtiene las dimensiones de entrada de una red a partir de las dimensiones de una imagen.
    
    :params img_shape: Dimensiones de la imagen en formato  CHW
    :return: Dimesiones de entrada de la red en formato NCHW
    """
    return tuple([None]+list(img_shape))

def make_new_shape_batch_tf(x):
    """Transforma un tf.tensor de formato NCHW a NHWC
    """
    return tf.transpose(x, [0, 2, 3, 1])

def make_new_shape_batch_np(x):
    """Transforma un np.array de imágenes de formato NCHW a NHWC
    """
    return np.transpose(x, [0, 2, 3, 1])

def norm_img(img):
    """Normaliza una imagen a valores [0,1]
    
    :param img: Imagen de entrada
    :return: Imagen normalizada al rango [0,1]
    """
    new_img = []
    for i in range(img.shape[-1]):
        tmp_img = img[:,:,i]
        tmp_img += abs(tmp_img.min())
        tmp_img /= tmp_img.max()
        new_img.append(new_img)
    return np.array(img)