""" An implementation of the Naive Bayes algorithm for a movie recommendation
system by scratch.


"""

import numpy as np

# The toy dataset we'll work with
X_train = np.array([
	[0, 1, 1],
	[0, 0, 1],
	[0, 0, 0],
	[1, 1, 0]
])

# The labels for our toy data
Y_train = ["Y", "N", "Y", "Y"]

# The new user
X_test = np.array([[1, 1, 0]])


# First group the data by label and record their indices by classes
def get_label_indices(labels):
	""" Group samples based on their labels and return indices.
	@param labels: the list of labels from our toy dataset
	@returns: dict, {class1: [indices], class2: [indices]}
	"""

	from collections import defaultdict

	label_indices = defaultdict(list)
	for index, label in enumerate(labels):
		label_indices[label].append(index)
	return label_indices


label_indices = get_label_indices(Y_train)
print("label_indices: \n", label_indices)


# With label_indices, we calculate the prior (initial probability of each
# class in the dataset before considering the evidence[features])
def get_prior(label_indices):
	"""
	Compute prior based on training samples.
	:param label_indices: grouped sample indices by class
	:return: dictionary, with class label as key, corresponding prior as the
	values.
	"""

	prior_ = {label: len(indices) for label, indices in label_indices.items()}
	total_count = sum(prior_.values())
	for label in prior_:
		prior_[label] /= total_count
	return prior_


prior = get_prior(label_indices)
print("Prior:", prior)

# We'll now calculate the likelihood, which is the conditional probability,
# P(feature|class)
def get_likelihood(features, label_indices, smoothing = 0):
	"""
	Compute the likelihood based on training samples.
	@param features: matrix of features
	@param label_indices: grouped sample indices by class
	@param smoothing: integer, audditive smoothing parameter
	@returns: dictionary, with class as key, corresponding conditional
	probability P(feature|class) vector as value
	"""
	likelihood = {}
	for label, indices in label_indices.items():
		likelihood[label] = features[indices, :].sum(axis=0) + smoothing
		total_count = len(indices)
		likelihood[label] = likelihood[label]/(total_count + 2 * smoothing)
	return likelihood


smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print("Likelihood: \n", likelihood)


# Computing the posterior for the testing/bew samples:
def get_posterior(X, prior, likelihood):
	"""
	Compute the posterior of testing samples, based on prior and likelihood
	@param X: testing samples
	@param prior: dictionary, with class labels as key, corresponding prior
	as the value
	@param likelihood: dictionary, with class labels as key, corresponding
	conditional probability vector as value
	@return: dictionary, with class label as key, corresponding posterior as
	value
	"""
	posteriors = []
	for x in X:
		# Posterior is proportional to prior * likelihood
		posterior = prior.copy()
		for label, likelihood_label in likelihood.items():
			for index, bool_value in enumerate(x):
				posterior[label] *= likelihood_label[index] if bool_value \
					else (1 - likelihood_label[index])
		# Normalize so they all sums up to 1
		sum_posterior = sum(posterior.values())
		for label in posterior:
			if posterior[label] == float("inf"):
				posterior[label] = 1.0
			else:
				posterior[label] /= sum_posterior
		posteriors.append(posterior.copy())
	return posteriors


# Predicting the class of one sample test using the prediction functions
posterior = get_posterior(X_test, prior, likelihood)
print("Posteriors: \n", posterior)

