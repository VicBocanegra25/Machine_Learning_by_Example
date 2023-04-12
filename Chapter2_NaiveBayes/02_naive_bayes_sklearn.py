"""An implementation of the Naive Bayes Algorithm using scikit-learn
Using the same toy data as in the file: 02_naive_bayes_scratch.py,
we'll calculate the posterior probabilities for a new person to like a new
film.

"""

from sklearn.naive_bayes import BernoulliNB
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

# Initializing a model with a smoothing factor of 1.0 and prior learned from
# the training set
clf = BernoulliNB(alpha = 1.0, fit_prior = True)

# Training the Naïve Bayes classifier with the fit method
clf.fit(X_train, Y_train)

# Obtaining the predicted probability results
pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities: \n', pred_prob)

# To directly acquire the predicted class (0.5 is the default threshold). If
# the predicted probability of class Y > 0.5, then Y is assigned else, N is used
pred = clf.predict(X_test)
print("[scikit-learn] Prediction:", pred)

