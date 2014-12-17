import theano as th
import theano.tensor as T

# unsupervised:
# sparse autoencoder: automatically learning features from unlabeled data
# http://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf

# Deep Neural Networks
# Sigmoid nonlinearity for hidden layers
# Softmax for the output layer
# Unsupervised learning (greedy layer-wise training)
# : maximize the lower-bound of the log-likelihood of the data (provide good initialization)
# supervised top-down training as final step (refine the features - intermediate features)
# : generative (up-down algorithm), discriminative (backpropogation)

# RBM
# http://www.pyimagesearch.com/2014/06/23/applying-deep-learning-rbm-mnist-using-python/
# http://scikit-learn.org/stable/modules/neural_networks.html
# http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
# http://www.cs.utoronto.ca/~hinton/absps/netflixICML.pdf
# http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/ml-practice/Introduction%20to%20Restricted%20Boltzmann%20Machines.ipynb
# http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

# https://www.kaggle.com/c/afsis-soil-properties/forums/t/10462/humorously-poor-results-pre-training-rbm-on-spectral-data-any-tips
# http://corpocrat.com/2014/10/17/machine-learning-using-restricted-boltzmann-machines/
# http://scikit-learn.org/stable/auto_examples/plot_rbm_logistic_classification.html

# http://hyperopt.github.io/hyperopt-sklearn/
# http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/ml-practice/Practice%20-%20how%20to%20train%20a%20deep%20pipeline.ipynb

# MDS
# http://scikit-learn.org/dev/modules/manifold.html#multi-dimensional-scaling-mds
# http://scikit-learn.org/dev/auto_examples/manifold/plot_mds.html


Alice: (Harry Potter = 1, Avatar = 1, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). Big SF/fantasy fan.
Bob: (Harry Potter = 1, Avatar = 0, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). SF/fantasy fan, but doesn't like Avatar.
Carol: (Harry Potter = 1, Avatar = 1, LOTR 3 = 1, Gladiator = 0, Titanic = 0, Glitter = 0). Big SF/fantasy fan.
David: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 1, Glitter = 0). Big Oscar winners fan.
Eric: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 1, Glitter = 0). Oscar winners fan, except for Titanic.
Fred: (Harry Potter = 0, Avatar = 0, LOTR 3 = 1, Gladiator = 1, Titanic = 1, Glitter = 0). Big Oscar winners fan.


#Oscar winners (containing LOTR 3, Gladiator, and Titanic)
#SF/fantasy (containing Harry Potter, Avatar, and LOTR 3)

# Gaussian-Bernoulli RBM for Real Valued Data Inputs
# https://github.com/benanne/morb/tree/master/morb
# https://www.cs.toronto.edu/~amnih/papers/rbmcf.pdf

# input data => normalize data
# bias unit (not connected to hidden node)
# hidden layer

# Feature Selection
# http://featureselection.asu.edu/software.php

# Drug RBM (DTI = Drug Target Interactions)
# http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694663/

class Model():
	def __init__(object, n_layers):
		self.n_layers = n_layers

	def activation_sigmoid(self, z):
		"""apply sigmoid activation, scale is 0:1"""
		return 1/(1+np.exp(-z))
		#T.nnet.sigmoid(x)

	def activation_sigmoidprime(self, z):
		"""apply derivative of sigmoid transfer function"""
		return activation_sigmoid(z) * (1 - activation_sigmoid(z))

	def activation_tanh(self, z):
		"""apply hyperbolic tangent activation function, scale is -1:1"""
		return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
		#T.tanh(x)

	def activation_tanhprime(self, z):
		"""apply derivative of tanh transfer function"""
		return 1 -(activation_sigmoid(z)**2)






#http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb





















import numpy as np
rng = np.random


def start_neural_network(samples):
	X = th.shared(samples)
	print X.get_value().shape
	print X.type()

	n_steps = 1000
	w = th.shared(rng.randn(100), name="w")
	b = th.shared(0., name="b")
	#print w.get_value(), b.get_value()

	th.printing.pydotprint(X)

	#http://nbviewer.ipython.org/github/philngo/whatsthatword/blob/5b0b00be3a5ab1ec05cef13dc089a77fd18f9ea2/ProcessBook.ipynb
	#http://www.nehalemlabs.net/prototype/blog/2013/10/17/solving-stochastic-differential-equations-with-theano/
	#http://erikbern.com/?cat=1
