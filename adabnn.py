#@title Dependencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import keras

from datetime import datetime

from keras import backend as K

from keras.models import Model

from keras.layers import Activation,Dense,Conv2D,Input,BatchNormalization,concatenate,Flatten

from keras.callbacks import EarlyStopping

from keras.initializers import Initializer

from keras.optimizers import SGD

from keras.constraints import max_norm

from keras.engine.topology import get_source_inputs

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape

from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import serialize_keras_object


from keras.engine import Layer
print('Last update: ', str(datetime.now()))


#@title Cost function
def F(y_true, y_pred):
    return K.mean(K.log( 1 + K.exp(1. - y_true * y_pred)), axis=-1)
print('Cost functions updated at: ', str(datetime.now()))

#@title Regularization function
lmbda = 10**-4
beta = 0

def RademacherComplexity(weight_matrix):
    M = weight_matrix.shape[0].value
    radNoise = K.variable([[random.choice([-1,1]) for k in range(M)]]).value()
    weight_matrix_sum = K.variable([K.sum(weight_matrix, axis=1)]).value()
    R = (1 / M) * K.dot(radNoise,K.transpose(weight_matrix_sum))
    R = (lmbda * R + beta) * (0.01 * K.sum(K.abs(weight_matrix)))
    return R
print('Regularization function updated at: ', str(datetime.now()))

#@title Initializer function
def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.
    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def sign_binarization(x):
    return K.sign(x)
    #if random.uniform(0,1) <= hard_sigmoid:
    #    return 1
    #else:
    #    return -1

def stochastic_binarization(x):
    hard_sigmoid = K.clip((x+1.)/2,0,1)
    tensor_bool = K.less_equal(K.random_uniform(shape=x.shape,minval=0.0,maxval=1.0),hard_sigmoid)
    tensor_float = K.cast(tensor_bool,dtype='float32')
    tensor_float_comp = tensor_float + K.constant(-1,shape=tensor_float.shape,dtype='float32')
    return tensor_float + tensor_float_comp
    #if random.uniform(0,1) <= hard_sigmoid:
    #    return 1
    #else:
    #    return -1

class BinaryUniform(Initializer):
    """Initializer that generates a binarized tensor from a uniform distribution.
    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=None):
    	value = K.random_uniform(shape, -1, 1,
    		                     dtype=dtype, seed=self.seed)
    	return sign_binarization(value)

    def get_config(self):
        return {
            'minval': self.minval,
            'maxval': self.maxval,
            'seed': self.seed,
        }


class BinaryGlorot_uniform(Initializer):
    """Binarized Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        Glorot & Bengio, AISTATS 2010
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self, scale=1.0,
                 mode='fan_avg',
                 distribution='uniform',
                 seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        scale /= max(1., float(fan_in + fan_out) / 2)
        limit = np.sqrt(3. * scale)
        value = K.random_uniform(shape, -limit, limit,
                                 dtype=dtype, seed=self.seed)
        return sign_binarization(value)

    def get_config(self):
        return {
            'scale': self.scale,
            'mode': self.mode,
            'distribution': self.distribution,
            'seed': self.seed
        }
print('Initializer functions updated at: ', str(datetime.now()))

#@title Activation functions
# Activations
def sign_binarization(x):
    return K.sign(x)
    #if random.uniform(0,1) <= hard_sigmoid:
    #    return 1
    #else:
    #    return -1

def stochastic_binarization(x):
    hard_sigmoid = K.clip((x+1.)/2,0,1)
    tensor_bool = K.less_equal(K.random_uniform(shape=x.shape,minval=0.0,maxval=1.1),hard_sigmoid)
    tensor_float = K.cast(tensor_bool,dtype='float32')
    tensor_float_comp = tensor_float + K.constant(-1,shape=tensor_float.shape,dtype='float32')
    return tensor_float + tensor_float_comp
    #if random.uniform(0,1) <= hard_sigmoid:
    #    return 1
    #else:
    #    return -1

def BinaryRelu(x, alpha=0., max_value=None):
    value = K.relu(x, alpha=alpha, max_value=max_value)
    return sign_binarization(value)
print('Activation functions updated at: ', str(datetime.now()))



#@title Declaring AdaBnn model
def AdaBnn(x_train,y_train,conf,classes=2,verbose=1):
    """Instatiates the AdaBnn architecture.

     # Arguments
        x_train: feature matrix for AdaNet framework.
        y_train: label matrix for AdaNet framework.
        conf: configuration parameters for adanet
            conf = dict({
                'network': {
                    'activation': 'relu',
                    'output_activation': 'sigmoid',
                    'optimizer': keras.optimizers.Adam(lr=0.0001),
                    'loss': 'binary_crossentropy',
                },
                'training':{
                    'batch_size': 32,
                    'epochs': 1,
                },
                'adanet':{
                    'B': 150,#3,
                    'T': 20#5
                    'delta': 1.01
                }
            })
        classes: number of classes to predict
        verbose: if 1: track the status of the training, 
            if 2: track also the timing for each part.
    # Returns
        A Keras model instance."""
    activation = conf['network']['activation']
    output_activation = conf['network']['output_activation']
    optimizer = conf['network']['optimizer']
    loss = conf['network']['loss']
    
    batch_size = conf['training']['batch_size']
    epochs = conf['training']['epochs']
    
    B = conf['adabnn']['B']
    T = conf['adabnn']['T']
    delta = conf['adabnn']['delta']
    seed = conf['adabnn']['seed']

    if classes == 2:
        output_shape = 1
    if classes > 2:
        output_shape = classes

    H = {}
    perf = []
    input_shape = x_train.shape[1::]
    inp = Input(shape=input_shape,name = 'input_layer')
    if len(input_shape) > 1:
        raise ValueError('The input data features'
                         'will be flatten into one dimension'
                         'next to the input layer')
        prep = Flatten(name='flat')(inp)
        h11 = Dense(B,
                    activation=activation,
                    kernel_regularizer=RademacherComplexity,
                    kernel_initializer=BinaryGlorot_uniform(seed=seed),
                    use_bias=False,
                    name = 'h11')(prep)
        H[0] = ['h11']
    else:
        h11 = Dense(B,
                    activation=activation,
                    kernel_regularizer=RademacherComplexity,
                    kernel_initializer=BinaryGlorot_uniform(seed=seed),
                    use_bias=False,
                    name = 'h11')(inp)
        H[0] = ['h11']
    out = Dense(output_shape,
                activation=output_activation,
                kernel_regularizer=RademacherComplexity,
                kernel_initializer=BinaryGlorot_uniform(seed=seed),
                use_bias=False,
                name = 'output_layer')(h11)
    model = Model(inp,out)
    depth = 1
    for t in range(0,T):
        H1 = H.copy()
        H2 = H.copy()
        if verbose in [1,2]: print('[INFO] Iteration: ',t + 1,'\n[INFO] Preparing H')
        for depth_h1 in range(1,depth+1):
            if depth_h1 == 1:
                h11_name = 'h' + str(depth_h1) + str(t+2)
                if verbose in [1,2]: print('[INFO] Adding layer node: ',h11_name)
                h1_layers = [Dense(B,
                                   activation=activation,
                                   kernel_regularizer=RademacherComplexity,
                                   use_bias=False,
                                   name=h11_name)(model.get_layer('input_layer').output)]
                H1[t+1] = [h11_name]
            else:
                conc = [model.get_layer(h).output for h in [h[depth_h1-2] for n,h in H.items() if len(h) >= (depth_h1-1)] ]
                conc.append(h1_layers[depth_h1-2])
                h22_name = 'h' + str(depth_h1) + str(t+2)
                if verbose in [1,2]: print('[INFO] Adding layer node: ',h22_name)
                h1_layers.append(Dense(B,
                                       activation=activation,
                                       kernel_regularizer=RademacherComplexity,
                                       kernel_initializer=BinaryGlorot_uniform(seed=seed),
                                       use_bias=False,
                                       name=h22_name)(concatenate(conc)))
                H1[t+1].append(h22_name)
        conc1 = [model.get_layer(y[-1]).output for (k,y) in H.items()]
        conc1.append(h1_layers[-1])
        out = Dense(output_shape,
                    activation=output_activation,
                    kernel_regularizer=RademacherComplexity,
                    kernel_initializer=BinaryGlorot_uniform(seed=seed),
                    use_bias=False,
                    name='output_layer')(concatenate(conc1))
        mo1 = Model(inputs=inp,
                    outputs=out)
        if verbose == 2:
            t1 = time.time()
            print('[TIMING] Defining layers and stacking them', t1 - t0)
        mo1.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])
        if verbose == 2:
            t2 = time.time()
            print('[TIMING] Compilation of the model', t2-t1)
        mo1.fit(x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=0)
        if verbose == 2:
            t3 = time.time()
            print('[TIMING] Fitting of the model', t3-t2)
        loss_h1,acc_h1 = mo1.evaluate(x_train,
                                      y_train,
                                      batch_size=batch_size,
                                      verbose=0)
        if verbose == 2:
            t4 = time.time()
            print('[TIMING] Evaluation of the model', t4-t3)

        if verbose in [1,2]: print('[INFO] Preparing Hprime')
        for depth_h2 in range(1,depth+2):
            if depth_h2 == 1:
                h11_name = 'h' + str(depth_h2) + str(t+2)
                if verbose in [1,2]: print('[INFO] Adding layer node: ',h11_name)
                h2_layers = [Dense(B,
                                   activation=activation,
                                   kernel_regularizer=RademacherComplexity,
                                   use_bias=False,
                                   name=h11_name)(model.get_layer('input_layer').output)]
                H2[t+1] = [h11_name]
            else:
                conc = [model.get_layer(h).output for h in [h[depth_h1-2] for n,h in H.items() if len(h) >= (depth_h1-1)] ]
                conc.append(h2_layers[depth_h2-2])
                h22_name = 'h' + str(depth_h2) + str(t+2)
                if verbose in [1,2]: print('[INFO] Adding layer node: ',h22_name)
                h2_layers.append(Dense(B,
                                       activation=activation,
                                       kernel_regularizer=RademacherComplexity,
                                       use_bias=False,
                                       name=h22_name)(concatenate(conc)))
                H2[t+1].append(h22_name)
        conc2 = [model.get_layer(y[-1]).output for (k,y) in H.items()]
        conc2.append(h2_layers[-1])
        out = Dense(output_shape,
                    activation=output_activation,
                    kernel_regularizer=RademacherComplexity,
                    kernel_initializer=BinaryGlorot_uniform(seed=seed),
                    use_bias=False,
                    name='output_layer')(concatenate(conc2))
        mo2 = Model(inputs=inp,
                    outputs=out)
        if verbose == 2:
            t5 = time.time()
            print('[TIMING] Defining layers and stacking them', t5 - t4)
        mo2.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])
        if verbose == 2:
            t6 = time.time()
            print('[TIMING] Compilation of the model', t6 - t5)
        mo2.fit(x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                verbose=0)
        if verbose == 2:
            t7 = time.time()
            print('[TIMING] Fitting of the model', t7 - t6)
        loss_h2,acc_h2 = mo2.evaluate(x_train,
                                      y_train,
                                      batch_size=batch_size,
                                      verbose=0)
        if verbose == 2:
            t8 = time.time()
            print('[TIMING] Evaluation of the model', t8 - t7)
        if loss_h1 < loss_h2:
            model = mo1
            H = H1
            loss_selected = loss_h1
            acc_selected = acc_h1
            if verbose in [1,2]: print('[INFO] H was selected')
        elif loss_h1 > loss_h2:
            if verbose in [1,2]: print('[INFO] Hprime was selected')
            model = mo2
            H = H2
            loss_selected = loss_h2
            acc_selected = acc_h2
            depth = depth + 1
        else:
            model = mo1
            H = H1
            loss_selected = loss_h1
            acc_selected = acc_h1
            if verbose in [1,2]: print('[INFO] Choosing H, without improving the loss function')
            break
        perf.append({
            "iteration": t + 1,
            "accuracy": acc_selected,
            "cost function": loss_selected
        })
    return model,perf

