# coding: utf-8

import numpy as np
import tensorflow as tf

wave = np.loadtxt('../data/12_29/wave.txt')
num_examples = wave.shape[0]
model_path = '../model/20190101-160833/arc_fault-30210'
with tf.Graph().as_default(), tf.device('/gpu:1'):
	graph = tf.get_default_graph()
	graph_file_name = model_path + '.meta'
	try:
		saver = tf.train.import_meta_graph(graph_file_name)
		x = graph.get_tensor_by_name('input:0')
		xl_1_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d/Squeeze:0')
		xh_1_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d_1/Squeeze:0')
		xl_2_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d_2/Squeeze:0')
		xh_2_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d_3/Squeeze:0')
		xl_3_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d_4/Squeeze:0')
		xh_3_op = graph.get_tensor_by_name('network/wavelet_cnn/average_pooling1d_5/Squeeze:0')

		tcn1_op = graph.get_tensor_by_name('network/tcn/tblock_0/conv1_prelu/add:0')
		tcn2_op = graph.get_tensor_by_name('network/tcn/tblock_1/conv1_prelu/add:0')
		tcn3_op = graph.get_tensor_by_name('network/tcn/tblock_2/conv1_prelu/add:0')
		se_op = graph.get_tensor_by_name('network/SENetLayer/mul:0')
		prelogits_op = graph.get_tensor_by_name('network/prelogits:0')
		is_training = graph.get_tensor_by_name('training:0')

		# weights
		wavelet_cnn_level1_kenerl_1_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d/kernel:0')
		wavelet_cnn_level1_kenerl_2_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d_1/kernel:0')
		wavelet_cnn_level2_kenerl_1_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d_2/kernel:0')
		wavelet_cnn_level2_kenerl_2_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d_3/kernel:0')
		wavelet_cnn_level3_kenerl_1_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d_4/kernel:0')
		wavelet_cnn_level3_kenerl_2_op = graph.get_tensor_by_name('network/wavelet_cnn/conv1d_5/kernel:0')

		tcn1_conv1_weight_op = graph.get_tensor_by_name('network/tcn/tblock_0/conv1/kernel:0')
		tcn1_conv2_weight_op = graph.get_tensor_by_name('network/tcn/tblock_0/conv2/kernel:0')

		config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
		sess = tf.Session(config=config)
		saver.restore(sess, model_path)

		xl_1, xh_1, xl_2, xh_2, xl_3, xh_3, tcn1, tcn2, tcn3, se, prelogits = sess.run(
			[xl_1_op, xh_1_op, xl_2_op, xh_2_op, xl_3_op, xh_3_op, tcn1_op, tcn2_op, tcn3_op, se_op, prelogits_op],
			feed_dict={x:wave.reshape(-1, 96, 1), is_training:False})
		wavelet_cnn_level1_kenerl_1, wavelet_cnn_level1_kenerl_2, \
		wavelet_cnn_level2_kenerl_1, wavelet_cnn_level2_kenerl_2, \
		wavelet_cnn_level3_kenerl_1, wavelet_cnn_level3_kenerl_2, \
		tcn1_conv1_weight, tcn1_conv2_weight = sess.run([wavelet_cnn_level1_kenerl_1_op, wavelet_cnn_level1_kenerl_2_op, 
			wavelet_cnn_level2_kenerl_1_op, wavelet_cnn_level2_kenerl_2_op, wavelet_cnn_level3_kenerl_1_op, wavelet_cnn_level3_kenerl_2_op,
			tcn1_conv1_weight_op, tcn1_conv2_weight_op])

		for i in range(num_examples):
			file_writer = open('result_{}.txt'.format(i), 'w')
			file_writer.write('xl_1\n')
			np.savetxt(file_writer, np.transpose(xl_1[i]), fmt='%.6f')
			file_writer.write('xh_1\n')
			np.savetxt(file_writer, np.transpose(xh_1[i]), fmt='%.6f')
			file_writer.write('xl_2\n')
			np.savetxt(file_writer, np.transpose(xl_2[i]), fmt='%.6f')
			file_writer.write('xh_2\n')
			np.savetxt(file_writer, np.transpose(xh_2[i]), fmt='%.6f')
			file_writer.write('xl_3\n')
			np.savetxt(file_writer, np.transpose(xl_3[i]), fmt='%.6f')
			file_writer.write('xh_3\n')
			np.savetxt(file_writer, np.transpose(xh_3[i]), fmt='%.6f')

			file_writer.write('tcn1\n')
			np.savetxt(file_writer, np.transpose(tcn1[i]), fmt='%.6f')
			file_writer.write('tcn2\n')
			np.savetxt(file_writer, np.transpose(tcn2[i]), fmt='%.6f')
			file_writer.write('tcn3\n')
			np.savetxt(file_writer, np.transpose(tcn3[i]), fmt='%.6f')
			file_writer.write('se\n')
			np.savetxt(file_writer, np.transpose(se[i]), fmt='%.6f')
			file_writer.write('prelogits\n')
			np.savetxt(file_writer, np.transpose(prelogits[i]), fmt='%.6f')
			file_writer.close()


		file_writer1 = open('wavelet_cnn_kenerl.txt', 'w')
		file_writer2 = open('tcn1_conv1_weight.txt', 'w')
		file_writer3 = open('tcn1_conv2_weight.txt', 'w')
		file_writer1.write('wavelet_cnn_level1_kenerl_1\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level1_kenerl_1[:, :, 0]), fmt='%.6f')
		file_writer1.write('wavelet_cnn_level1_kenerl_2\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level1_kenerl_2[:, :, 0]), fmt='%.6f')
		file_writer1.write('wavelet_cnn_level2_kenerl_1\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level2_kenerl_1[:, :, 0]), fmt='%.6f')
		file_writer1.write('wavelet_cnn_level2_kenerl_2\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level2_kenerl_2[:, :, 0]), fmt='%.6f')
		file_writer1.write('wavelet_cnn_level3_kenerl_1\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level3_kenerl_1[:, :, 0]), fmt='%.6f')
		file_writer1.write('wavelet_cnn_level3_kenerl_2\n')
		np.savetxt(file_writer1, np.transpose(wavelet_cnn_level3_kenerl_2[:, :, 0]), fmt='%.6f')
		for i in range(25):
			np.savetxt(file_writer2, np.transpose(tcn1_conv1_weight[:,:,i]), fmt='%.6f')
			np.savetxt(file_writer3, np.transpose(tcn1_conv2_weight[:,:,i]), fmt='%.6f')
		file_writer1.close()
		file_writer2.close()
		file_writer3.close()
		
	except Exception as e:
		raise Exception(e)
	finally:
		sess.close()