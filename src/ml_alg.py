from sklearn import metrics
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import pandas as pd
from pprint import pprint

from sklearn.neural_network import BernoulliRBM
import rbm as rbm

from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances


# determine if this should be normal PCA vs RandomPCA
def pca_reduction(X, pca):
	X_r = pca.fit(X).transform(X)
	# percentage of variance explained by each of selected components
	print"Variance Ratio: \n{0}".format(pca.explained_variance_ratio_[:15])
	# components with maximum variance
	# print"Components...: \n{0}".format(pca.components_[:1][0][:10])
	return pca, X_r


def rbm_dnn_scikit(training_data, n_hidden):
	rbm = BernoulliRBM(n_components=n_hidden, n_iter=5000)
	rbm.fit(training_data)
	# Compute the hidden layer activation probabilities, P(h=1|v=X) : Latent representations of the data
	# Mappings.shape = (n_samples, n_components), training_data.shape = (n_samples, n_features); n_features =n_visible
	mappings = rbm.transform(training_data)
	print "n_samples:{0}, n_visible={1}, n_hidden={2}".format(training_data.shape[0], training_data.shape[1], mappings.shape[1])
	# Weight Matrix : components.shape = (n_components, n_features)
	print rbm.components_.shape, rbm.components_.T.shape 
	pprint(rbm.components_.T)


def rbm_dnn(training_data, n_visible, n_hidden, visible_labels):
	# keeping max_epochs low for large samples, local machine processing
	r = rbm.RBM(num_visible = n_visible, num_hidden = n_hidden)
	r.train(training_data, max_epochs = 10)
	
	rows = ["Bias Unit"] + visible_labels
	hidden_labels = ["Hidden " + str(i+1) for i in range(n_hidden)]
	cols = ["Bias Unit"] + hidden_labels
	mappings = pd.DataFrame(r.weights, index=rows, columns=cols)

	# calculate simalarites of drugs across all hidden nodes
	dists = cosine_similarity(mappings)
	dists = pd.DataFrame(dists, columns=mappings.index)
	dists.index = dists.columns

	# Dimensionality Reduction	
	# pca, X_r = pca_reduction(mappings.values, PCA())
	# plot_pca_comp_analysis(pca)

	return mappings, dists, r

def rbm_dnn_run(r, test_data, n_visible, n_hidden, visible_labels):
	# See what hidden units are activated
	d = r.run_visible(test_data) 

	rows = ["Bias Unit"] + visible_labels
	hidden_labels = ["Hidden " + str(i+1) for i in range(n_hidden)]
	cols = ["Bias Unit"] + hidden_labels
	mappings = pd.DataFrame(r.weights, index=rows, columns=cols)

	# calculate simalarites of drugs across all hidden nodes
	dists = cosine_similarity(mappings)
	dists = pd.DataFrame(dists, columns=mappings.index)
	dists.index = dists.columns

	return mappings, dists, r



def get_similar(items, dists, n=None):
	# calculates which items are most similar to the input that was provided
	items = [item for item in items if item in dists.columns]
	items_summed = dists[items].apply(lambda row: np.sum(row), axis=1)
	items_summed = items_summed.order(ascending=False)
	ranked_items = items_summed.index[items_summed.index.isin(items)==False]
	ranked_items = ranked_items.tolist()
	if n is None: return ranked_items
	else: return ranked_items[:n]


def plot_pca_comp_analysis(pca_v):
    comp_id = [i + 1 for i in range(len(pca_v.explained_variance_ratio_))]             
    fig = plt.figure(figsize=(8,5))
    plt.plot(comp_id, pca_v.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlim(0,25)
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance')
    plt.savefig('../plots/pca_analysis.png')
    
    #plt.show()

def plot_heat_map(data, x_labels, y_labels, title, image_name):
	# Plot a heatmap representation
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)
	fig = plt.gcf()
	fig.set_size_inches(40,30)
	# turn off the frame
	ax.set_frame_on(False)
	# put the major ticks at the middle of each cell
	ax.xaxis.tick_top()
	ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
	ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
	for label in ax.get_xticklabels(): label.set_fontsize(24) 
	for label in ax.get_yticklabels(): label.set_fontsize(24) 
	# want a more natural, table-like display
	ax.invert_yaxis()
	# Set the labels
	ax.set_xticklabels(x_labels, minor=False) 
	ax.set_yticklabels(y_labels, minor=False)
	# rotate the labels
	plt.xticks(rotation=90)
	ax.grid(True)
	# title
	# plt.title(title)
	#plt.figtext(.02, .02, title, size='x-large')
	# Turn off all the ticks
	plt.tight_layout()

	for t in ax.xaxis.get_major_ticks(): 
	    t.tick1On = False 
	    t.tick2On = False 
	for t in ax.yaxis.get_major_ticks(): 
	    t.tick1On = False 
	    t.tick2On = False  

	plt.savefig('../plots/' + image_name + '.png')    
	
	#plt.show()


