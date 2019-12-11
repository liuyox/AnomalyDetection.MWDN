# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import tensorflow as tf
import argparse
#import matplotlib.pyplot as plt
from load_data import load_data
from utils import Logger
from sklearn.model_selection import KFold
from sklearn import metrics 
from scipy import interpolate

def main(args):
	dataset = load_data(args.filename)
	with tf.Graph().as_default(), tf.device('/gpu:1'):
		graph = tf.get_default_graph()
		graph_file_name = args.model_path + '.meta'
		try:
			saver = tf.train.import_meta_graph(graph_file_name)
			x = graph.get_tensor_by_name('input:0')
			y = graph.get_tensor_by_name('y_pred:0')
			is_training = graph.get_tensor_by_name('training:0')

			config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
			sess = tf.Session(config=config)
			saver.restore(sess, args.model_path)

			num_batches = int(np.ceil(dataset.num_examples / args.batch_size))
			prob = np.zeros((num_batches*args.batch_size,))
			label = np.zeros((num_batches*args.batch_size,))
			for i in range(num_batches):
				feature, label[i*args.batch_size:(i+1)*args.batch_size] = dataset.next_batch(args.batch_size)
				feature = np.reshape(feature, (args.batch_size, -1, 1))
				temp = sess.run(y, feed_dict={x:feature, is_training:False})
				prob[i*args.batch_size:(i+1)*args.batch_size] = temp[:,1]
		except Exception as e:
			raise Exception(e)
		finally:
			sess.close()

	thresholds = np.arange(0, 1, 0.001)
	tpr, fpr, accuracy, best_threshold_index = calc_roc(thresholds,prob,label)

	return tpr, fpr, accuracy, best_threshold_index
	
def calc_roc(thresholds,prob,label,num_folds=10):
    assert(prob.shape == label.shape)

    num_examples = prob.shape[0]
    num_thresholds = thresholds.shape[0]

    k_fold = KFold(n_splits=num_folds,shuffle=False)

    tprs = np.zeros((num_folds,num_thresholds))
    fprs = np.zeros((num_folds,num_thresholds))
    accuracy = np.zeros(num_folds)

    indices = np.arange(num_examples)

    for fold_index, (train_index, test_index) in enumerate(k_fold.split(indices)):
        accuracy_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _,_,accuracy_train[threshold_index] = calc_accuracy(prob[train_index],label[train_index],threshold)
        best_threshold_index = np.argmax(accuracy_train)
        tmp_accuracy = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            tprs[fold_index,threshold_index], fprs[fold_index, threshold_index], tmp_accuracy[threshold_index] = calc_accuracy(prob[test_index],label[test_index],threshold)
        accuracy[fold_index] = tmp_accuracy[best_threshold_index]
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy, best_threshold_index

def calc_accuracy(prob, label, threshold):
	label = label.astype(bool)
	pred_label = np.greater_equal(prob, threshold)
	tp = np.sum(np.logical_and(pred_label,label))
	fp = np.sum(np.logical_and(pred_label,np.logical_not(label)))
	tn = np.sum(np.logical_and(np.logical_not(pred_label),np.logical_not(label)))
	fn = np.sum(np.logical_and(np.logical_not(pred_label),label))

	if tp + fn == 0:
	    tpr = 0
	else:
	    tpr = tp / (tp + fn)

	if fp + tn == 0:
	    fpr = 0
	else:
	    fpr = fp / (fp + tn)

	accuracy = (tp + tn) / prob.size

	return tpr, fpr, accuracy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--filename', type=str, default='./data/test.txt', help='')
	parser.add_argument('--model_path', type=str, default='./model/20180924-121216/model-240', help='')
	parser.add_argument('--batch_size', type=int, default=50, help='')

	args = parser.parse_args()
	tpr, fpr, accuracy, best_threshold_index = main(args)

	auc = metrics.auc(fpr, tpr)
	thresholds = np.arange(0, 1, 0.001)
	print('auc: %s'%auc)
	print('accuracy: %s'%np.mean(accuracy))
	print('best_threshold: %s'%thresholds[best_threshold_index])
	print('tpr: %s'%tpr[best_threshold_index])
	print('fpr: %s'%fpr[best_threshold_index])

	#plt.figure()
	#plt.plot(fpr, tpr, color='b',label='ROC (AUC = %0.2f)' %auc,lw=2, alpha=.8)
	#plt.xlabel('False Positive Rate')
	#plt.ylabel('True Positive Rate')
	#plt.show()

	# save
	file_name = 'ABCD-E'
	data = np.column_stack((tpr, fpr))
	np.savetxt(file_name+'.txt', data, fmt='%f')

	with open(file_name+'.log', 'w') as file_writer:
		file_writer.write('auc: %s\n'%auc)
		file_writer.write('accuracy: %s\n'%np.mean(accuracy))
		file_writer.write('best_threshold: %s\n'%thresholds[best_threshold_index])
		file_writer.write('tpr: %s\n'%tpr[best_threshold_index])
		file_writer.write('fpr: %s\n'%fpr[best_threshold_index])
