#coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class DataSet(object):
    def __init__(self,feature,label):
        assert feature.shape[0] == label.shape[0],(
        'feature.shape:%s label.shape:%s'%(feature.shape,label.shape))
        self._num_examples = feature.shape[0]
        self._feature = feature
        self._label = label
        self._index_in_epoch = 0
        
    @property
    def num_examples(self):
        return self._num_examples
        
    @property
    def feature(self):
        return self._feature
        
    @property
    def label(self):
        return self._label
        
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start==0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._feature = self._feature[perm]
            self._label = self._label[perm]
        if start+batch_size>self._num_examples:
            rest_num_examples = self._num_examples - start
            feature_rest_part = self._feature[start:self._num_examples]
            label_rest_part = self._label[start:self._num_examples]

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._feature = self._feature[perm]
            self._label = self._label[perm]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            feature_new_part = self._feature[start:end]
            label_new_part = self._label[start:end]
            return np.concatenate((feature_rest_part,feature_new_part),axis=0),np.concatenate((label_rest_part,label_new_part),axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._feature[start:end],self._label[start:end]

def load_data(file_name):
    data = np.loadtxt(file_name)
    np.random.shuffle(data)
    dataset = DataSet(data[:,:-1],data[:,-1].astype(int))
    return dataset
