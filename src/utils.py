#coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import logging
import os
import tensorflow as tf
from six import iteritems

class Logger(object):
    def __init__(self, logger_name=None, file_name=None, control_log=True, log_level='INFO'):
        LOG_FORMAT = "%(asctime)s - [%(levelname)s]: %(message)s"
        # DATE_FORMAT = None  # "%Y-%d-%m %H:%M:%S"
        if log_level == 'DEBUG':
            level = logging.DEBUG
        elif log_level == 'INFO':
            level = logging.INFO
        elif log_level == 'ERROR':
            level = logging.ERROR
        elif log_level == 'WARNING':
            level = logging.WARNING
        else:
            raise ValueError('Invalid log level [%s]' % log_level)

        # create logger
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT)

        if control_log:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

        if file_name:
            file_handler = logging.FileHandler(file_name)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self._logger.warn(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

def save_arguments(args,file_name):
    with open(file_name,'w') as file:
        for key, value in iteritems(vars(args)):
            file.write('%s:\t%s\n'%(key,value))

def center_loss(features, label, num_class, alpha):
    with tf.variable_scope('center_loss'):
        num_features = features.get_shape().as_list()[1]
        centers = tf.get_variable('centers',[num_class,num_features],dtype=tf.float32,initializer=tf.constant_initializer(0),trainable=False)
        label = tf.reshape(label,[-1])
        centers_batch = tf.gather(centers,label)

        loss = tf.reduce_mean(tf.norm(features - centers_batch, ord=2, axis=-1))

        diff = centers_batch - features

        unique_label, unique_index, unique_count = tf.unique_with_counts(label)
        appear_times = tf.gather(unique_count, unique_index)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = diff / tf.cast((1+appear_times), tf.float32)
        diff = alpha * diff

        centers_update_op = tf.scatter_sub(centers, label, diff)
    return loss, centers, centers_update_op

def focal_loss(labels, logits, alpha, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    with tf.variable_scope('focal_loss'):
        y_pred = tf.nn.softmax(logits, dim=-1) # [batch_size,num_classes]
        labels = tf.one_hot(labels, depth = y_pred.shape[1])
        loss = -alpha * labels * ((1-y_pred)**gamma) * tf.log(y_pred)
        loss = tf.reduce_sum(loss,axis=1)
    return tf.reduce_mean(loss)