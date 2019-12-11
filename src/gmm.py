# coding: utf-8

import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import interpolate
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm


def select_gmm(train_data, n_components_range, cv_types):
	bic = []
	lowest_bic = np.infty
	for cv_type in cv_types:
		for n_components in n_components_range:
			gmm = GaussianMixture(n_components=n_components,
				covariance_type=cv_type).fit(negative_examples)
			bic.append(gmm.bic(negative_examples))
			if bic[-1] < lowest_bic:
				lowest_bic = bic[-1]
				best_gmm = gmm

	bic = np.array(bic)
	return bic, best_gmm


def plot_bars(bic, n_components_range, cv_types):
	# Plot the BIC scores
	bars = []
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
	                              'darkorange'])
	plt.figure(figsize=(8, 6))
	spl = plt.subplot(1, 1, 1)
	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
	    xpos = np.array(n_components_range) + .2 * (i - 2)
	    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
	                                  (i + 1) * len(n_components_range)],
	                        width=.2, color=color))
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
	    .2 * np.floor(bic.argmin() / len(n_components_range))
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)
	plt.show()


def calc_roc(thresholds, prob, label, num_folds=10):
    assert(prob.shape == label.shape)

    num_examples = prob.shape[0]
    num_thresholds = thresholds.shape[0]

    k_fold = KFold(n_splits=num_folds, shuffle=True)

    tprs = np.zeros((num_folds, num_thresholds))
    fprs = np.zeros((num_folds, num_thresholds))
    F1s = np.zeros((num_folds, num_thresholds))
    accuracy = np.zeros(num_folds)

    indices = np.arange(num_examples)

    for fold_index, (train_index, test_index) in enumerate(k_fold.split(indices)):
        g_means_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, g_means_train[threshold_index] = calc_accuracy(
                prob[train_index], label[train_index], threshold)
        best_threshold_index = np.argmax(g_means_train)
        tmp_accuracy = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            tprs[fold_index, threshold_index], fprs[fold_index, threshold_index], tmp_accuracy[threshold_index], F1s[fold_index,
                threshold_index], _ = calc_accuracy(prob[test_index], label[test_index], threshold)
        accuracy[fold_index] = tmp_accuracy[best_threshold_index]
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    F1 = np.mean(F1s, 0)
    return tpr, fpr, F1, accuracy, best_threshold_index


def calc_accuracy(prob, label, threshold):
	label = label.astype(bool)
	pred_label = np.greater_equal(prob, threshold)
	tp = np.sum(np.logical_and(pred_label, label))
	fp = np.sum(np.logical_and(pred_label, np.logical_not(label)))
	tn = np.sum(np.logical_and(np.logical_not(pred_label), np.logical_not(label)))
	fn = np.sum(np.logical_and(np.logical_not(pred_label), label))

	if tp + fn == 0:
	    tpr = 0
	else:
	    tpr = tp / (tp + fn)

	if fp + tn == 0:
	    fpr = 0
	else:
	    fpr = fp / (fp + tn)

	if tp + fp == 0:
		precision = 0
	else:
		precision = tp / (tp + fp)

	accuracy = (tp + tn) / prob.size
	F1 = 2 * precision * tpr / (precision + tpr)
	g_means = np.sqrt(tpr * (1-fpr))
	return tpr, fpr, accuracy, F1, g_means

def dimension_reduction(dataset, n_components):
	features = dataset[:, 0:-1]
	label = dataset[:, -1]
	pca = PCA(n_components=n_components).fit(features)
	new_features = pca.transform(features)
	return np.column_stack((new_features, label))

def make_ellipses(gmm, ax, colors):
	for n, color in enumerate(colors):
		covariances = gmm.covariances_[n][:2, :2]
		v, w = np.linalg.eigh(covariances)
		u = w[0] / np.linalg.norm(w[0])
		angle = np.arctan2(u[1], u[0])
		angle = 180 * angle / np.pi  # convert to degrees
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
		                          180 + angle, color=color)
		ell.set_clip_box(ax.bbox)
		ell.set_alpha(0.5)
		ax.add_artist(ell)

if __name__ == '__main__':
	dataset = np.loadtxt('../data/12_29/train_features_25.txt')
	np.random.shuffle(dataset)
	features = dataset[:, 0:-1]
	label = dataset[:, -1].astype(int)

	positive_indexs = np.where(label==1)[0]
	negative_indexs = np.where(label==0)[0]

	positive_examples = features[positive_indexs]
	negative_examples = features[negative_indexs]


	#n_components_range = range(1, 11)
	#cv_types = ['spherical', 'tied', 'diag', 'full']
	#bic, clf = select_gmm(negative_examples, n_components_range, cv_types)
	
	#plot_bars(bic, n_components_range, cv_types)

	n_components = 9
	clf = GaussianMixture(n_components=n_components,
				covariance_type='full').fit(negative_examples)

	test_data = np.loadtxt('../data/12_29/test_features_25.txt')
	label = test_data[:,-1]
	
	

	# plt.figure()
	# pos_scores = clf.score_samples(positive_examples)
	# neg_scores = clf.score_samples(negative_examples)
	# plt.plot(pos_scores)
	# plt.plot(neg_scores)
	# plt.show()

	# scores = np.concatenate((pos_scores, neg_scores))
	# label = np.concatenate((label[positive_indexs], label[negative_indexs]))
	scores = clf.score_samples(test_data[:,:-1])
	scores = (scores - min(scores)) / (max(scores) - min(scores))
	#np.savetxt('../result/scores.txt', scores, fmt='%.6f')
	#exit(0)

	thresholds = np.arange(scores.min(), scores.max(), 0.001)
	tpr, fpr, F1, accuracy, best_threshold_index = calc_roc(thresholds, scores, np.logical_not(label))

	auc = metrics.auc(fpr, tpr)
	print('auc: %s'%auc)
	print('accuracy: %s'%np.mean(accuracy))
	print('best_threshold: %s'%thresholds[best_threshold_index])
	print('tpr: %s'%tpr[best_threshold_index])
	print('fpr: %s'%fpr[best_threshold_index])
	print('F1: %s'%F1[best_threshold_index])

	np.savetxt('../result/arc_roc.txt', np.column_stack((tpr, fpr)), fmt='%f')

	plt.figure()
	plt.plot(fpr, tpr, color='b',label='ROC (AUC = %0.2f)' %auc,lw=2, alpha=.8)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()


	# online learning
	for n_iter in range(100):
		log_prob_norm, log_resp = clf._e_step(test_data[label==0,:-1])
		clf._m_step(test_data[label==0,:-1], log_resp)


	scores = clf.score_samples(test_data[:,:-1])
	scores = (scores - min(scores)) / (max(scores) - min(scores))
	np.savetxt('../result/scores.txt', scores, fmt='%.6f')
	exit(0)

	thresholds = np.arange(scores.min(), scores.max(), 0.001)
	tpr, fpr, F1, accuracy, best_threshold_index = calc_roc(thresholds, scores, np.logical_not(label))

	auc = metrics.auc(fpr, tpr)
	print('auc: %s'%auc)
	print('accuracy: %s'%np.mean(accuracy))
	print('best_threshold: %s'%thresholds[best_threshold_index])
	print('tpr: %s'%tpr[best_threshold_index])
	print('fpr: %s'%fpr[best_threshold_index])
	print('F1: %s'%F1[best_threshold_index])

	np.savetxt('../result/arc_roc.txt', np.column_stack((tpr, fpr)), fmt='%f')

	plt.figure()
	plt.plot(fpr, tpr, color='b',label='ROC (AUC = %0.2f)' %auc,lw=2, alpha=.8)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()

	'''

	clf, log_likelyhood = GaussianMixture(n_components=9,
				covariance_type='full', verbose=2, verbose_interval=1, max_iter=200).fit(negative_examples)
	print(log_likelyhood)
	np.savetxt('log_likelyhood.txt', log_likelyhood, fmt='%.6f')
	exit(0)
	fig = plt.figure()
	ax = plt.subplot(111)
	make_ellipses(clf, ax, colors = ['m', 'c', 'y','g','r'])
	np.random.shuffle(negative_examples)
	for point in negative_examples[0:1000]:
		color = 'b'
		plt.scatter(point[0], point[1], 30,c=color,marker='o',linewidths=0.75,edgecolors='k')
	fig.set_size_inches(4, 4)
	plt.savefig('../result/gmm_figure_arc.svg', format='svg')
	plt.show()

	x_min = min(negative_examples[:,0])
	x_max = max(negative_examples[:,0])
	y_min = min(negative_examples[:,1])
	y_max = max(negative_examples[:,1])

	x = np.linspace(x_min, x_max, 100)
	y = np.linspace(y_min, y_max, 100)
	X, Y = np.meshgrid(x, y)
	XX = np.array([X.ravel(), Y.ravel()]).T
	temp = clf.predict_proba(XX)
	Z = np.zeros_like(X, dtype=np.float64)
	for i in range(3):
		Z += temp[:,i].reshape(X.shape)
	CS = plt.contour(X, Y, Z,norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
	CB = plt.colorbar(CS, shrink=0.8, extend='both')
	plt.scatter(negative_examples[:, 0], negative_examples[:, 1], .8)

	plt.title('Negative log-likelihood predicted by a GMM')
	plt.axis('tight')
	plt.show()
	np.savetxt('Z.txt', Z)

	'''