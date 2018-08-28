# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

#
#    Keras
#
def augment_image(img, img_gen=None, crop=None):
    """Función que realiza el data-augmentation. Devuelve imágenes en orden aleatorio y recortadas de forma aleatoria dentro de unas dimensiones especificadas.
    
    :param img: Batch de imágenes a procesar.
    :param img_gen: Función que se ocupa del procesamiento de las imágenes.
    :param crop: Dimensiones a las que recortar las imágenes.
    :return: Imagen aumentada.
    """
    if img_gen == None and crop == None:
        return img

    for x_b, _ in img_gen.flow(np.asarray([img], dtype=img.dtype), np.asarray([0], dtype="int32")):
        if crop != None:
            img_size = x_b[0].shape[-1]
            #Recorte aleatorio de las imágenes
            x_start = np.random.randint(0, img_size-crop+1)
            y_start = np.random.randint(0, img_size-crop+1)
            ret = x_b[0][:, y_start:y_start+crop, x_start:x_start+crop]

            return ret
        break
    return x_b[0]

def _get_slices(length, batch_size):
    """Obtiene los valores de imagenes que comprende un batch.
    """
    slices, b = [], 0

    while True:
        if b*batch_size >= length:
            break
        slices.append( slice(b*batch_size, (b+1)*batch_size) )
        b += 1

    return slices

def _dataset_generator(x, y, batch_size, num_classes, rnd_state=None):
    """Generador que devuelve los batch de imágenes.
    
    :param x: Patrones de entrada.
    :param y: Clases a las que pertenecen los patrones.
    :param batch_size: Tamaño de los batch a devolver.
    :param num_classes:  Número de clases del problema.
    :param rnd_state: Estado aleatorio para mezclar las imágenes.
    :return: Batch de imágenes aumentadas junto con las clases.
    """
    while True:
        assert x.shape[0] == y.shape[0]
        slices = _get_slices(x.shape[0], batch_size)

        if rnd_state != None:
            rnd_state.shuffle(slices)

        for i in slices:
            x_, y_ = x[i], y[i]
            images_x = [ augment_image(img, img_gen, crop) for img in x_ ]
            images_x = np.asarray(images_x, dtype="float32")
            yield images_x, y_

def dataset_generator(img_gen=None, crop=None):
    """Función que devuelve un generador de imágenes aumentadas.
    
    :param img_gen: Función que realiza el procesamiento de las imágenes.
    :param crop: Dimensiones a las que recortar las imágenes.
    :return: :func:`_dataset_generator`
    """

    dat_gen = _dataset_generator

    return dat_gen

#
#    Tensorflow
#

#Invert NCHW image
def flip_horz(image):
    """Reimplementación de la función que voltea imágenes horizontalmente para usar el formato NCHW en tensorflow.
    """
    with tf.name_scope("random_flip") as scope:
        uniform_random = tf.random_uniform([], 0, 1)
        mirror_cond = tf.less(uniform_random, .5)
        result = tf.cond(mirror_cond,
                         lambda: tf.reverse(image, [2]),
                         lambda: image)
        return result

#The desired preprocessing
def train_preprocess(image, label, num_labels=8):
    """Función que implementa el preprocesamiento requerido en entrenamiento
    """
    with tf.name_scope("train_preprocessing") as scope:
        image = tf.random_crop(image, [3, 224, 224])
        #En el data-augmentation solo hace horizontal_flip
        image = flip_horz(image)
        label = tf.one_hot(tf.cast(label, tf.int32), num_labels)
        return image, label

def test_preprocess(image, label, num_labels=8):
    """Función que implementa el preprocesamiento requerido en test
    """
    with tf.name_scope("test_preprocessing") as scope:
        #Central crop image
        image = image[:,16:240,16:240]
        label = tf.one_hot(tf.cast(label, tf.int32), num_labels)
        return image, label