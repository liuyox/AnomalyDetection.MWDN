# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
import numpy as np
import argparse
import utils
from model import Network
from load_data import load_data
from tqdm import tqdm
from sklearn import preprocessing
from datetime import datetime


def train(args):
    sub_dir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, sub_dir)
    model_dir = os.path.join(args.model_dir, sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_logger = utils.Logger('train', file_name=os.path.join(
        log_dir, 'train.log'), control_log=False)
    test_logger = utils.Logger(
        'test', file_name=os.path.join(log_dir, 'test.log'))

    utils.save_arguments(args, os.path.join(log_dir, 'arguments.txt'))

    # data
    split_dataset(args.dataset_path, args.train_ratio, args.pos_ratio)
    base_dir = os.path.dirname(args.dataset_path)
    train_dataset = load_data(os.path.join(base_dir, 'train.txt'))
    test_dataset = load_data(os.path.join(base_dir, 'test.txt'))

    dataset_size = train_dataset.num_examples
    train_logger.info('dataset size: %s' % dataset_size)

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.device('/gpu:3'):
        tf.set_random_seed(10)
        x = tf.placeholder(tf.float32, shape=[None, args.seq_len, 1], name='input')
        y = tf.placeholder(tf.int64, shape=[None], name='label')
        one_hot_y = tf.one_hot(y, depth=2, dtype=tf.int64)
        is_training = tf.placeholder(tf.bool, name='training')

        net = Network(args.seq_len, args.tcn_channels, args.tcn_kernel_size, 
            args.tcn_dropout, args.embedding_size, args.num_classes, args.weight_decay)
        prelogits, logits, embeddings = net(x, is_training)

        # accuracy
        with tf.variable_scope('metrics'):
            tpr_op, fpr_op, g_mean_op, accuracy_op, f1_op = calc_accuracy(logits, y)

        # loss
        with tf.variable_scope('loss'):
            focal_loss_op = utils.focal_loss(y, logits, 5.0)
            cross_entropy_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=one_hot_y), name='cross_entropy')
            center_loss_op, centers, centers_update_op = utils.center_loss(
                prelogits, y, args.num_classes, args.center_loss_alpha)
            regularization_loss_op = tf.reduce_sum(tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
            loss_op = center_loss_op * args.center_loss_factor + \
                cross_entropy_op + regularization_loss_op# + L_loss_op * args.mwdn_loss_factor[0] + H_loss_op * args.mwdn_loss_factor[1]

        # optimizer
        with tf.variable_scope('optimizer'), tf.control_dependencies([centers_update_op]):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # optimizer_op = tf.train.MomentumOptimizer(learning_rate_op, 0.9, name='optimizer')
            optimizer = tf.train.AdamOptimizer(args.lr, name='optimizer')
            train_op = optimizer.minimize(loss_op, global_step)

        # summary
        tf.summary.scalar('total_loss', loss_op)
        tf.summary.scalar('l2_loss', regularization_loss_op)
        tf.summary.scalar('cross_entropy', cross_entropy_op)
        tf.summary.scalar('center_loss', center_loss_op)
        tf.summary.scalar('focal_loss', focal_loss_op)
        #tf.summary.scalar('l_loss', L_loss_op)
        #tf.summary.scalar('h_loss', H_loss_op)
        tf.summary.scalar('accuracy', accuracy_op)
        tf.summary.scalar('tpr', tpr_op)
        tf.summary.scalar('fpr', fpr_op)
        tf.summary.scalar('g_mean', g_mean_op)
        tf.summary.scalar('f1', f1_op)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=100)
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            tf.global_variables_initializer().run()

            if args.pretrained_model:
                ckpt = tf.train.get_checkpoint_state(args.pretrained_model)
                saver.restore(sess, ckpt.model_checkpoint_path)

            steps_per_epoch = np.ceil(dataset_size / args.batch_size).astype(int)
            batch_num_seq = range(steps_per_epoch)
            best_test_accuracy = 0.0
            best_test_fpr = 1.0
            try:
                for epoch in range(1, args.max_epochs+1):
                    batch_num_seq = tqdm(
                        batch_num_seq, desc='Epoch: {:d}'.format(epoch), ascii=True)
                    for step in batch_num_seq:
                        feature, label = train_dataset.next_batch(args.batch_size)
                        feature = np.reshape(feature, (args.batch_size, args.seq_len, 1))
                        if step % args.display == 0:
                            tensor_list = [train_op, summary_op, global_step, accuracy_op, tpr_op,
                                fpr_op, g_mean_op, cross_entropy_op, center_loss_op, focal_loss_op]
                            _, summary, train_step, accuracy, tpr, fpr, g_mean, cross_entropy, center_loss, focal_loss = sess.run(
                                tensor_list, feed_dict={x: feature, y: label, is_training: True})
                            train_logger.info('Train Step: %d, accuracy: %.3f%%, tpr: %.3f%%, fpr: %.3f%%, g_mean: %.3f%%, cross_entropy: %.4f, center_loss: %.4f, focal_loss: %.4f'
                                % (train_step, accuracy*100, tpr*100, fpr*100, g_mean*100, cross_entropy, center_loss, focal_loss))
                        else:
                            _, summary, train_step = sess.run([train_op, summary_op, global_step], feed_dict={
                                                              x: feature, y: label, is_training: True})
                        writer.add_summary(summary, global_step=train_step)
                        writer.flush()

                    # evaluate
                    num_batches = int(np.ceil(test_dataset.num_examples / args.batch_size))
                    accuracy_array = np.zeros((num_batches,), np.float32)
                    tpr_array = np.zeros((num_batches,), np.float32)
                    fpr_array = np.zeros((num_batches,), np.float32)
                    g_mean_array = np.zeros((num_batches,), np.float32)
                    for i in range(num_batches):
                        feature, label = test_dataset.next_batch(args.batch_size)
                        feature = np.reshape(feature, (args.batch_size, args.seq_len, 1))
                        tensor_list = [accuracy_op, tpr_op, fpr_op, g_mean_op]
                        feed_dict = {x: feature, y: label, is_training: False}
                        accuracy_array[i], tpr_array[i], fpr_array[i], g_mean_array[i] = sess.run(
                            tensor_list, feed_dict=feed_dict)
                    test_logger.info('Validation Epoch: %d, train_step: %d, accuracy: %.3f%%, tpr: %.3f%%, fpr: %.3f%%, g_mean: %.3f%%'
                        % (epoch, train_step, np.mean(accuracy_array)*100, np.mean(tpr_array)*100, np.mean(fpr_array)*100, np.mean(g_mean_array)*100))

                    test_accuracy = np.mean(accuracy_array)
                    test_fpr = np.mean(fpr_array)
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        saver.save(sess, os.path.join(model_dir, 'arc_fault'),
                                   global_step=train_step)
                    elif test_accuracy == best_test_accuracy and test_fpr < best_test_fpr:
                        best_test_fpr = test_fpr
                        saver.save(sess, os.path.join(model_dir, 'arc_fault'),
                                   global_step=train_step)

            except Exception as e:
                train_logger.error(e)
            writer.close()


def split_dataset(dataset_path, train_ratio, pos_ratio):
    data = np.loadtxt(dataset_path)

    pos_index = np.where(np.equal(data[:, -1], 1))[0]
    neg_index = np.where(np.equal(data[:, -1], 0))[0]

    num_examples = data.shape[0]
    num_train = int(np.ceil(num_examples * train_ratio))

    num_pos = int(np.round(num_train * pos_ratio))
    num_neg = num_train - num_pos
    if num_pos > pos_index.shape[0] * 0.95:
        train_pos_index = np.random.choice(pos_index, num_pos, replace=True)
    else:
        train_pos_index = np.random.choice(pos_index, num_pos, replace=False)

    if num_neg > neg_index.shape[0] * 0.95:
        train_neg_index = np.random.choice(neg_index, num_neg, replace=True)
    else:
        train_neg_index = np.random.choice(neg_index, num_neg, replace=False)

    train_examples = np.row_stack((data[train_pos_index], data[train_neg_index]))
    test_examples = np.delete(data, np.concatenate(
        (np.unique(train_pos_index), np.unique(train_neg_index)), axis=0), axis=0)

    # normalize
    scaler = preprocessing.Normalizer(norm='l2').fit(train_examples[:, :-1])

    train_feature = scaler.transform(train_examples[:, :-1])
    train_label = train_examples[:, -1]

    test_feature = scaler.transform(test_examples[:, :-1])
    test_label = test_examples[:, -1]

    base_dir = os.path.dirname(dataset_path)
    file_name = os.path.join(base_dir, 'train.txt')
    np.savetxt(file_name, np.column_stack(
        (train_feature, train_label)), fmt='%.6f')

    file_name = os.path.join(base_dir, 'test.txt')
    np.savetxt(file_name, np.column_stack((test_feature, test_label)), fmt='%.6f')


def calc_accuracy(logtis, label):
    label_ = tf.cast(label, tf.bool)
    logtis_ = tf.cast(tf.argmax(logtis, axis=1), tf.bool)

    TP = tf.reduce_sum(tf.cast(tf.logical_and(label_, logtis_), tf.int64))
    FP = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.logical_not(label_), logtis_), tf.int64))
    TN = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.logical_not(label_), tf.logical_not(logtis_)), tf.int64))
    FN = tf.reduce_sum(tf.cast(tf.logical_and(
        label_, tf.logical_not(logtis_)), tf.int64))

    TPR = tf.where(tf.equal(TP + FN, 0), tf.constant(0.0,
                   dtype=tf.float64), (TP / (TP + FN)))
    TNR = tf.where(tf.equal(TN + FP, 0), tf.constant(0.0, dtype=tf.float64), (TN / (TN + FP)))
    FPR = tf.where(tf.equal(TN + FP, 0), tf.constant(0.0, dtype=tf.float64), (FP / (TN + FP)))
    F1 = tf.where(tf.equal(TP + FP, 0), tf.constant(0.0, dtype=tf.float64), (TP / (TP + FP)))

    G = tf.sqrt(TPR*TNR)
    accuray = (TP + TN) / (TP + TN + FP + FN)
    return TPR, FPR, G, accuray, F1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/data_96.txt', help='')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='')
    parser.add_argument('--pos_ratio', type=float, default=0.5, help='')

    parser.add_argument('--model_dir', type=str, default='../model', help='')
    parser.add_argument('--log_dir', type=str, default='../log', help='')

    parser.add_argument('--seq_len', type=int, default=96, help='')
    parser.add_argument('--num_classes', type=int, default=2, help='')
    parser.add_argument('--embedding_size', type=int, default=25, help='')
    parser.add_argument('--tcn_channels', type=list, default=[25, 25, 25], help='')
    parser.add_argument('--tcn_kernel_size', type=int, default=4, help='')
    parser.add_argument('--tcn_dropout', type=float, default=0.1, help='')

    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--max_epochs', type=int, default=100, help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--display', type=int, default=100, help='')

    parser.add_argument('--center_loss_factor',type=float,default=0.05, help='center loss ratio in total loss.')
    parser.add_argument('--center_loss_alpha',type=float,default=0.5,help='center update rate for center loss.')

    parser.add_argument('--mwdn_loss_factor', type=list, default=[3.0, 3.0], help='')

    parser.add_argument('--pretrained_model', type=str, default=None, help='')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #设置需要使用的GPU的编号

    train(args)
