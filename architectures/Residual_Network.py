# -*- coding: utf-8 -*-

from keras.regularizers import l2

from keras.models import Model
from keras.initializers import Constant, he_normal
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D,\
                         Dense, Flatten, Activation, ZeroPadding2D,\
                         Add, Activation, BatchNormalization

#from reimplementations import he_normal_mine as he_normal
from unimodal_extensions import _add_pois, _add_binom

def _residual_block(layer, n_out_channels, stride=1, nonlinearity='relu'):
    """Crea un bloque residual de la red.
    
    :param layer: Capa de anterior al bloque residual a crear.
    :param n_out_channels: Número de filtros deseados para las convoluciones realizadas en el bloque.
    :param stride: Stride para la primera convolución realizada y en el caso de ser mayor a 1, el usado en un primer AveragePooling.
    :param nonlinearity: No linealidad aplicada a las salidas de las BatchNormalization aplicadas.
    :return: La última capa del bloque residual.
    """
    conv = layer
    if stride > 1:
        #padding: https://stackoverflow.com/a/47213171
        layer = AveragePooling2D(pool_size=1, strides=stride, padding="same")(layer)
    #Si no hay concordancia de dimensiones entre las capas se hace un padding con ceros
    if (n_out_channels != int(layer.get_shape()[1])):
        diff = n_out_channels - int(layer.get_shape()[1])
        diff_2 = int(diff / 2)
        if diff %2 == 0:
            width_tp = ((diff_2, diff_2),(0,0))
        else:
            width_tp = (((diff_2) + 1, diff_2),(0,0))
        # Para que el pad se haga en la dimension correcta, al no poder seleccionar
        #como en lasagne batch_ndim, se usa data_format='channels_last'
        layer = ZeroPadding2D(padding=(width_tp), data_format='channels_last')(layer)
    conv = Conv2D(filters=n_out_channels, kernel_size=(3,3),
                  strides=(stride,stride), padding='same',
                  activation='linear',
                  kernel_initializer=he_normal(),
                  bias_initializer=Constant(0.),
                  kernel_regularizer=l2(1e-4),
                  bias_regularizer=l2(1e-4))(conv)
    conv = BatchNormalization(beta_initializer=Constant(0.),
                              gamma_initializer=Constant(1.),
                              beta_regularizer=l2(1e-4),
                              gamma_regularizer=l2(1e-4))(conv)
    conv = Activation(nonlinearity)(conv)
    conv = Conv2D(filters=n_out_channels, kernel_size=(3,3),
                  strides=(1,1), padding='same',
                  activation='linear',
                  kernel_initializer=he_normal(),
                  bias_initializer=Constant(0.),
                  kernel_regularizer=l2(1e-4),
                  bias_regularizer=l2(1e-4))(conv)
    conv = BatchNormalization(beta_initializer=Constant(0.),
                              gamma_initializer=Constant(1.),
                              beta_regularizer=l2(1e-4),
                              gamma_regularizer=l2(1e-4))(conv)
    sum_ = Add()([conv, layer])
    return Activation(nonlinearity)(sum_)


def _resnet_2x4(l_in, nf=[32, 64, 128, 256], N=2):
    """Función para crear de manera automática los múltiples bloques residuales de la red.
    
    :param l_in: Capa de entrada.
    :param nf: Lista con el distinto número de filtros a utilizar por capa.
    :param N: Profundidad de los bloques residuales intermedios y contiguos.
    :return: La última capa después de los bloques residuales.
    """
    assert len(nf) == 4 # this is a 4-block resnet
    layer = Conv2D(filters=nf[0], kernel_size=7,
                   strides=2, activation='relu',
                   bias_initializer=Constant(0),
                   padding='same', kernel_initializer=he_normal(),
                   kernel_regularizer=l2(1e-4),
                   bias_regularizer=l2(1e-4))(l_in)
    layer = MaxPooling2D(pool_size=3, strides=2)(layer)

    #Residual blocks go here
    #
    for i in range(N):
        layer = _residual_block(layer, nf[0], prefix="a%i" % i)
    layer = _residual_block(layer, nf[1], prefix="aa", stride=2)

    for i in range(N):
        layer = _residual_block(layer, nf[1], prefix="b%i" % i)
    layer = _residual_block(layer, nf[2], prefix="bb%", stride=2)

    for i in range(N):
        layer = _residual_block(layer, nf[2], prefix="c%i" % i)
    layer = _residual_block(layer, nf[3], prefix="cc", stride=2)

    for i in range(N):
        layer = _residual_block(layer, nf[3], prefix="dd%i" % i)

    layer = AveragePooling2D(pool_size=int(layer.get_shape()[-1]), strides=1, padding='valid')(layer)
    
    layer = Flatten()(layer)

    return layer

class Resnet_2x4():
    """Construye la arquitectura base de la red residual usada.
    
    :param input_shape: Dimensiones de las imagenes usadas (NxWxH).
    :param num_labels: Número de clases del problema tratado.
    """
    def __init__(self, input_shape=(3,224,224), num_labels=8):
        self.inputs = Input(shape=input_shape)
        self.net = _resnet_2x4(self.inputs)
        self.pred = Dense(units=num_labels, activation='softmax')(self.net)

    def get_net(self):
        """Devuelve la salida de la red para los patrones dados como entrada.
        
        :return: Clasificación de la red para los patrones de entrada.
        """
        return self.pred

class Resnet_2x4_poisson():
    """Construye la arquitectura de la red residual usada añadiendo al final las capas unimodales con la distribución de Poisson.
    
    :param tau_mode: Modo de ejecución para el parámetro tau, es decir, si se aprende como peso de la red o se toma constante el valor pasado.
    :type tau_mode: "non_learnable" or "sigm_learnable"
    :param input_shape: Dimensiones de las imagenes usadas (NxWxH).
    :param num_labels: Número de clases del problema tratado.
    :param tau: Valor inicial de tau.
    """
    def __init__(self, tau_mode, input_shape=(3,224,224), num_labels=8, tau=1.):
        assert tau_mode in ["non_learnable", "sigm_learnable"]

        self.inputs = Input(shape=input_shape)
        self.net = _resnet_2x4(self.inputs)
        self.pred = _add_pois(self.net, end_nonlinearity='softplus',
                              num_classes=num_labels, tau=tau,
                              tau_mode=tau_mode)


    def get_net(self):
        """Devuelve la salida de la red para los patrones dados como entrada.
        
        :return: Clasificación de la red para los patrones de entrada.
        """
        return self.pred

class Resnet_2x4_binomial():
    """Construye la arquitectura de la red residual usada añadiendo al final las capas unimodales con la distribución de Poisson.
    
    :param tau_mode: Modo de ejecución para el parámetro tau, es decir, si se aprende como peso de la red o se toma constante el valor pasado.
    :type tau_mode: "non_learnable" or "sigm_learnable"
    :param input_shape: Dimensiones de las imagenes usadas (NxWxH).
    :param num_labels: Número de clases del problema tratado.
    :param tau: Valor inicial de tau.
    """
    def __init__(self, tau_mode, input_shape=(3,224,224), num_labels=8, tau=1.):
        assert tau_mode in ["non_learnable", "sigm_learnable"]

        self.inputs = Input(shape=input_shape)
        self.net = _resnet_2x4(self.inputs)
        self.pred = _add_binom(self.net, num_classes=num_labels,
                               tau=tau, tau_mode=tau_mode)

    def get_net(self):
        """Devuelve la salida de la red para los patrones dados como entrada.
        
        :return: Clasificación de la red para los patrones de entrada.
        """
        return self.pred
