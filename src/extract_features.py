# coding: utf-8

import tensorflow as tf
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_data import load_data
from tqdm import tqdm

def extract_features(data_file_name, model_path, batch_size, embedding_size):
	dataset = load_data(data_file_name)
	with tf.Graph().as_default(), tf.device('/gpu:1'):
		graph = tf.get_default_graph()
		graph_file_name = model_path + '.meta'
		try:
			saver = tf.train.import_meta_graph(graph_file_name)
			x = graph.get_tensor_by_name('input:0')
			y = graph.get_tensor_by_name('network/prelogits:0')
			is_training = graph.get_tensor_by_name('training:0')

			config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
			sess = tf.Session(config=config)
			saver.restore(sess, model_path)

			num_batches = int(np.ceil(dataset.num_examples / batch_size))
			embeddings = np.zeros((num_batches*batch_size, embedding_size))
			label = np.zeros((num_batches*batch_size,))

			batch_num_seq = range(num_batches)
			batch_num_seq = tqdm(batch_num_seq,desc=None,ascii=True)
			for i in batch_num_seq:
				features, label[i*batch_size:(i+1)*batch_size] = dataset.next_batch(batch_size)
				features = np.reshape(features, (batch_size, -1, 1))
				temp = sess.run(y, feed_dict={x:features, is_training:False})
				embeddings[i*batch_size:(i+1)*batch_size] = temp
		except Exception as e:
			raise Exception(e)
		finally:
			sess.close()
	return embeddings, label

if __name__ == '__main__':
	embeddings, label = extract_features(data_file_name='../data/12_29/test.txt', model_path='../model/20190101-160833/arc_fault-72345', 
		batch_size=100, embedding_size=25)

	np.savetxt('../data/12_29/test_features_25.txt', np.column_stack((embeddings, label)), fmt='%.6f')

	'''
	dataset = np.loadtxt('../data/test_features2.txt')
	np.random.shuffle(dataset)
	features = dataset[:, 0:-1]
	label = dataset[:, -1].astype(int)

	plt.figure()

	for index, feature in enumerate(features[0:1000]):
		color = 'b'
		marker = 'o'
		if label[index] == 1:
			#continue
			color = 'r'
			marker = '^'
		plt.scatter(feature[0], feature[1], c=color, marker=marker)

	plt.show()
	'''