#coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

def prelu(x, name=None):
    with tf.variable_scope(name, 'prelu'):
        i = int(x.get_shape()[-1])
        alpha = tf.get_variable('alpha',
                                shape=(i,),
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.25))
        y = tf.nn.relu(x) + tf.multiply(alpha, -tf.nn.relu(-x))
    return y

class CausalConv1d(tf.layers.Conv1D):
    def __init__(self,
            filters,
            kernel_size,
            strides=1,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, **kwargs
            )

    def call(self, x):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        x = tf.pad(x, tf.constant([[0,0],[padding,0],[0,0]],dtype=tf.int32))
        return super().call(x)

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
        trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)
        
        self._n_outputs = n_outputs
        self._kernel_size = kernel_size
        self._strides = strides
        self._dilation_rate = dilation_rate
        self._dropout = dropout

    def build(self, input_shape):
        self._conv1 = CausalConv1d(self._n_outputs, self._kernel_size, strides=self._strides,
            dilation_rate=self._dilation_rate, activation=None, name='conv1')
        self._conv2 = CausalConv1d(self._n_outputs, self._kernel_size, strides=self._strides,
            dilation_rate=self._dilation_rate, activation=None, name='conv2')

        self._dropout1 = tf.layers.Dropout(self._dropout, [tf.constant(1), tf.constant(1), tf.constant(self._n_outputs)])
        self._dropout2 = tf.layers.Dropout(self._dropout, [tf.constant(1), tf.constant(1), tf.constant(self._n_outputs)])

        if input_shape[2] != self._n_outputs:
            #self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self._down_sample = tf.layers.Dense(self._n_outputs, activation=None)
        else:
            self._down_sample = None

    def call(self, x, training=True):
        y = self._conv1(x)
        y = tf.contrib.layers.layer_norm(y)
        y = prelu(y, name='conv1_prelu')
        #y = tf.nn.elu(y)
        y = self._dropout1(y, training=training)

        y = self._conv2(y)
        y = self._dropout2(y, training=training)

        if self._down_sample is not None:
            x = self._down_sample(x)
        y = tf.contrib.layers.layer_norm(x+y)
        return prelu(y, name='conv2_prelu')

class TemporalConvNet(tf.layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
        trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)
        self._layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self._layers.append(TemporalBlock(out_channels, kernel_size, strides=1, 
                dilation_rate=dilation_size, dropout=dropout, name='tblock_{}'.format(i)))

    def call(self, x, training=True):
        y = x
        for layer in self._layers:
            y = layer(y, training=training)
        return y

class SENetLayer(tf.layers.Layer):
    def __init__(self, out_dim, ratio, trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer, 
            name=name, **kwargs)
        self._out_dim = out_dim
        self._ratio = ratio


    def build(self, input_shape):
        self._squeeze = tf.layers.AveragePooling1D((input_shape[1],), 1, name='squeeze')
        self._excitation_1 = tf.layers.Dense(self._out_dim // self._ratio, activation=prelu, name='excitation_1')
        self._excitation_2 = tf.layers.Dense(self._out_dim, activation=tf.nn.sigmoid, name='excitation_2')

    def call(self, x):
        squeeze = self._squeeze(x)
        excitation = self._excitation_1(squeeze)
        excitation = self._excitation_2(excitation)
        excitation = tf.reshape(excitation, [-1, 1, self._out_dim])
        scale = x * excitation
        return scale

def attentionBlock(x, counters, dropout):
    """self attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
        counters: to keep track of names
        dropout: add dropout after attention
    # Returns
        A tensor of shape [N, L, Cin]
    """

    k_size = x.get_shape()[-1].value
    v_size = x.get_shape()[-1].value

    name = get_name('attention_block', counters)
    with tf.variable_scope(name):
        # [N, L, k_size]
        key = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 0.01))
        key = tf.nn.dropout(key, 1.0 - dropout)
        # [N, L, k_size]
        query = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        query = tf.nn.dropout(query, 1.0 - dropout)
        value = tf.layers.dense(x, units=v_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        value = tf.nn.dropout(value, 1.0 - dropout)

        logits = tf.matmul(query, key, transpose_b=True)
        logits = logits / np.sqrt(k_size)
        weights = tf.nn.softmax(logits, name="attention_weights")
        output = tf.matmul(weights, value)

    return output


class Wavelet_CNN(tf.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)

        # wavelet filters
        self._l_filter = np.array([-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304], dtype=np.float32)
        self._h_filter = np.array([-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106], dtype=np.float32)

    def build(self, input_shape):
        # mWDN layers
        self._mWDN_1L = tf.layers.Conv1D(1, self._l_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._l_filter))
        self._mWDN_1H = tf.layers.Conv1D(1, self._h_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._h_filter))

        self._mWDN_2L = tf.layers.Conv1D(1, self._l_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._l_filter))
        self._mWDN_2H = tf.layers.Conv1D(1, self._h_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._h_filter))

        self._mWDN_3L = tf.layers.Conv1D(1, self._l_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._l_filter))
        self._mWDN_3H = tf.layers.Conv1D(1, self._h_filter.size, strides=1, padding='valid',
            kernel_initializer=tf.constant_initializer(self._h_filter))

    def call(self, x):
        # 1 level
        x1 = tf.pad(x, tf.constant([[0,0],[0, self._l_filter.size - 1],[0,0]],dtype=tf.int32))
        al_1 = self._mWDN_1L(x1)
        al_1 = tf.contrib.layers.layer_norm(al_1)
        #al_1 = tf.nn.elu(al_1)
        al_1 = prelu(al_1, 'level1_prelu1')

        ah_1 = self._mWDN_1H(x1)
        ah_1 = tf.contrib.layers.layer_norm(ah_1)
        #ah_1 = tf.nn.elu(ah_1)
        ah_1 = prelu(ah_1, 'level1_prelu2')

        xl_1 = tf.layers.average_pooling1d(al_1, 2, 2)
        xh_1 = tf.layers.average_pooling1d(ah_1, 2, 2)

        # 2 level
        x2 = tf.pad(xl_1, tf.constant([[0,0],[0, self._l_filter.size - 1],[0,0]],dtype=tf.int32))
        al_2 = self._mWDN_2L(x2)
        al_2 = tf.contrib.layers.layer_norm(al_2)
        #al_2 = tf.nn.elu(al_2)
        al_2 = prelu(al_2, 'level2_prelu1')

        ah_2 = self._mWDN_2H(x2)
        ah_2 = tf.contrib.layers.layer_norm(ah_2)
        #ah_2 = tf.nn.elu(ah_2)
        ah_2 = prelu(ah_2, 'level2_prelu2')

        xl_2 = tf.layers.average_pooling1d(al_2, 2, 2)
        xh_2 = tf.layers.average_pooling1d(ah_2, 2, 2)

        # 3 level
        x3 = tf.pad(xl_2, tf.constant([[0,0],[0, self._l_filter.size - 1],[0,0]],dtype=tf.int32))
        al_3 = self._mWDN_3L(x3)
        al_3 = tf.contrib.layers.layer_norm(al_3)
        #al_3 = tf.nn.elu(al_3)
        al_3 = prelu(al_3, 'level3_prelu1')

        ah_3 = self._mWDN_3H(x3)
        ah_3 = tf.contrib.layers.layer_norm(ah_3)
        #ah_3 = tf.nn.elu(ah_3)
        ah_3 = prelu(ah_3, 'level3_prelu2')

        xl_3 = tf.layers.average_pooling1d(al_3, 2, 2)
        xh_3 = tf.layers.average_pooling1d(ah_3, 2, 2)

        return xl_1, xh_1, xl_2, xh_2, xl_3, xh_3

class MWDN(tf.layers.Layer):
    def __init__(self, seq_len, trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=trainable, dtype=dtype, activity_regularizer=activity_regularizer,
            name=name, **kwargs)

        self._seq_len = seq_len

        # wavelet filters
        self._l_filter = np.array([-0.0106,0.0329,0.0308,-0.187,-0.028,0.6309,0.7148,0.2304])
        self._h_filter = np.array([-0.2304,0.7148,-0.6309,-0.028,0.187,0.0308,-0.0329,-0.0106])

    def build(self, input_shape):
        # mWDN layers
        self._mWDN_1L = tf.layers.Dense(self._seq_len, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len, self._l_filter)))
        self._mWDN_1H = tf.layers.Dense(self._seq_len, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len, self._h_filter)))

        self._mWDN_2L = tf.layers.Dense(self._seq_len // 2, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len//2, self._l_filter)))
        self._mWDN_2H = tf.layers.Dense(self._seq_len // 2, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len//2, self._h_filter)))

        self._mWDN_3L = tf.layers.Dense(self._seq_len // 4, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len//4, self._l_filter)))
        self._mWDN_3H = tf.layers.Dense(self._seq_len // 4, activation=None,
            kernel_initializer=tf.constant_initializer(self._create_weight(self._seq_len//4, self._h_filter)))

    def _create_weight(self, shape, kernel, is_comp=False):
        max_epsilon = np.min(np.abs(kernel))
        if is_comp:
            weights = np.zeros((shape, shape), dtype=np.float32)
        else:
            weights = np.random.randn(shape, shape) * 0.01 * max_epsilon
            weights = weights.astype(np.float32)

        for i in range(0, shape):
            index = 0
            for j in range(i, shape):
                if index < kernel.size:
                    weights[j, i] = kernel[index]
                    index += 1
        return weights

    def call(self, x):
        # 1 level
        al_1 = self._mWDN_1L(tf.reshape(x, [-1, self._seq_len]))
        #al_1 = tf.nn.sigmoid(al_1)
        al_1 = tf.contrib.layers.layer_norm(al_1)
        al_1 = prelu(al_1, 'level1_prelu1')

        ah_1 = self._mWDN_1H(tf.reshape(x, [-1, self._seq_len]))
        #ah_1 = tf.nn.sigmoid(ah_1)
        ah_1 = tf.contrib.layers.layer_norm(ah_1)
        ah_1 = prelu(ah_1, 'level1_prelu2')

        xl_1 = tf.layers.average_pooling1d(tf.reshape(al_1, [-1, self._seq_len, 1]), 2, 2, name='xl_1')
        xh_1 = tf.layers.average_pooling1d(tf.reshape(ah_1, [-1, self._seq_len, 1]), 2, 2, name='xh_1')

        # 2 level
        al_2 = self._mWDN_2L(tf.reshape(xl_1, [-1, self._seq_len//2]))
        #al_2 = tf.nn.sigmoid(al_2)
        al_2 = tf.contrib.layers.layer_norm(al_2)
        al_2 = prelu(al_2, 'level2_prelu1')

        ah_2 = self._mWDN_2H(tf.reshape(xl_1, [-1, self._seq_len//2]))
        #ah_2 = tf.nn.sigmoid(ah_2)
        ah_2 = tf.contrib.layers.layer_norm(ah_2)
        ah_2 = prelu(ah_2, 'level2_prelu2')

        xl_2 = tf.layers.average_pooling1d(tf.reshape(al_2, [-1, self._seq_len//2, 1]), 2, 2, name='xl_2')
        xh_2 = tf.layers.average_pooling1d(tf.reshape(ah_2, [-1, self._seq_len//2, 1]), 2, 2, name='xh_2')

        # 3 level
        al_3 = self._mWDN_3L(tf.reshape(xl_2, [-1, self._seq_len//4]))
        #al_3 = tf.nn.sigmoid(al_3)
        al_3 = tf.contrib.layers.layer_norm(al_3)
        al_3 = prelu(al_3, 'level3_prelu1')

        ah_3 = self._mWDN_3H(tf.reshape(xl_2, [-1, self._seq_len//4]))
        #ah_3 = tf.nn.sigmoid(ah_3)
        ah_3 = tf.contrib.layers.layer_norm(ah_3)
        ah_3 = prelu(ah_3, 'level3_prelu2')

        xl_3 = tf.layers.average_pooling1d(tf.reshape(al_3, [-1, self._seq_len//4, 1]), 2, 2, name='xl_3')
        xh_3 = tf.layers.average_pooling1d(tf.reshape(ah_3, [-1, self._seq_len//4, 1]), 2, 2, name='xh_3')

        cmp_mWDN1_L = tf.Variable(self._create_weight(self._seq_len, self._l_filter, True))
        cmp_mWDN1_H = tf.Variable(self._create_weight(self._seq_len, self._h_filter, True))

        cmp_mWDN2_L = tf.Variable(self._create_weight(self._seq_len // 2, self._l_filter, True))
        cmp_mWDN2_H = tf.Variable(self._create_weight(self._seq_len // 2, self._h_filter, True))

        cmp_mWDN3_L = tf.Variable(self._create_weight(self._seq_len // 4, self._l_filter, True))
        cmp_mWDN3_H = tf.Variable(self._create_weight(self._seq_len // 4, self._h_filter, True))

        L_loss = tf.norm((self._mWDN_1L.kernel - cmp_mWDN1_L), 2) + \
        tf.norm((self._mWDN_2L.kernel - cmp_mWDN2_L), 2) + \
        tf.norm((self._mWDN_3L.kernel - cmp_mWDN3_L), 2)

        H_loss = tf.norm((self._mWDN_1H.kernel - cmp_mWDN1_H), 2) + \
        tf.norm((self._mWDN_2H.kernel - cmp_mWDN2_H), 2) + \
        tf.norm((self._mWDN_3H.kernel - cmp_mWDN3_H), 2)

        return xl_1, xh_1, xl_2, xh_2, xl_3, xh_3, L_loss, H_loss

class Network(tf.layers.Layer):
    def __init__(self, seq_len,tcn_channels, tcn_kernel_size, tcn_dropout, embedding_size, num_classes, weight_decay,
        trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs):
        super().__init__(trainable=True, name=None, dtype=None, activity_regularizer=None, **kwargs)
        self._seq_len = seq_len
        self._tcn_channels = tcn_channels
        self._tcn_kernel_size = tcn_kernel_size
        self._tcn_dropout = tcn_dropout
        self._embedding_size = embedding_size
        self._num_classes = num_classes
        self._weight_decay = weight_decay

    def build(self, input_shape):
        # wavelet cnn
        self._wavelet_cnn = Wavelet_CNN(name='wavelet_cnn')
        self._mwdn = MWDN(self._seq_len, name='mwdn')

        # temporal convolutional networks
        self._tcn = TemporalConvNet(self._tcn_channels, self._tcn_kernel_size, self._tcn_dropout, name='tcn')
        self._se_layer = SENetLayer(self._tcn_channels[-1], 4, name='SENetLayer')
        # embedding layer
        #self._embedding = tf.layers.Dense(self._embedding_size, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #    kernel_regularizer=None, name='bottleneck')
        self._softmax_layer = tf.layers.Dense(self._num_classes, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_regularizer=None, name='logits')

    def call(self, x, training=True):

        xl_1, xh_1, xl_2, xh_2, xl_3, xh_3 = self._wavelet_cnn(x)
        #xl_1, xh_1, xl_2, xh_2, xl_3, xh_3, L_loss, H_loss = self._mwdn(x)

        x1 = xh_1
        x2 = xh_2
        #x1 = tf.concat([xl_1, xh_1], axis=-1)
        #x2 = tf.concat([xl_2, xh_2], axis=-1)
        x3 = tf.concat([xl_3, xh_3], axis=-1)

        width = x1.shape.as_list()[1]

        # upsampling
        x2 = tf.expand_dims(x2, axis=1)
        x2 = tf.image.resize_images(x2, [1, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x2 = tf.squeeze(x2, axis=1)

        x3 = tf.expand_dims(x3, axis=1)
        x3 = tf.image.resize_images(x3, [1, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x3 = tf.squeeze(x3, axis=1)

        concat_x = tf.concat([x1, x2, x3], axis=-1)
        y = self._tcn(concat_x, training)
        y = self._se_layer(y)

        #y1 = self._tcn_1(x1, training)
        #y2 = self._tcn_2(x2, training)
        #y3 = self._tcn_3(x3, training)

        #y = tf.concat([y1[:,-1,:], y2[:,-1,:], y3[:,-1,:]], axis=1)
 

        #y = self._tcn_1(x, training)
        #y = tf.layers.flatten(y)

        '''
        shape = y.shape.as_list()
        y = tf.layers.average_pooling1d(y, shape[1], strides=1)
        y = tf.layers.flatten(y)
        prelogits = self._embedding(y)
        '''
        #prelogits = self._embedding(y)
        prelogits = tf.squeeze(tf.layers.average_pooling1d(y, y.shape.as_list()[1], strides=1), axis=1, name='prelogits')
        logits = self._softmax_layer(prelogits)

        embeddings = tf.nn.l2_normalize(prelogits, axis=1, name='embeddings')

        return prelogits, logits, embeddings#, L_loss, H_loss
