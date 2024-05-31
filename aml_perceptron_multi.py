
"""This file shows a couple of implementations of the perceptron learning
algorithm. It is based on the code from Lecture 3, but using the slightly
more compact perceptron formulation that we saw in Lecture 6.

There are two versions: Perceptron, which uses normal NumPy vectors and
matrices, and SparsePerceptron, which uses sparse vectors and matrices.
The latter may be faster when we have high-dimensional feature representations
with a lot of zeros, such as when we are using a "bag of words" representation
of documents.
"""

import numpy as np
from sklearn.base import BaseEstimator
import random
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X): 
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w) # w is a matrix, one row of weights for each class

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the class with the highest score
        out = np.argmax(scores)

        return out

        
class SVCMulticlass(LinearClassifier):
    """
    Implementing the SVC algorithm for assignment PA4 
    """

    def __init__(self, epochs=20, lambda_=1e-5):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        s#elf.find_classes(Y) # remove this??

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1] # number of cols
        self.w = np.zeros(n_features)

        # SVC algorithm

        t = 0 #used iteratively in eta calc
        xy_pairs = list(zip(X, Ye))
        n_iter = X.shape[0] * self.epochs
        
        for i in range(n_iter):

            #select random pair from xy_pairs, get x and y
            (x, y) = random.choice(xy_pairs)
            t += 1
            eta = 1 / (self.lambda_ * t)

            # Compute the output score for this instance
            score = x.dot(self.w)

            loss_gradient = np.zeros_like(self.w)
            
            for j in range(n_labels):
                if j != y_true:
                    margin = 1 - scores[j] + scores[y_true]
                    if margin > 0:
                        loss_gradient[:, j] -= x
                        loss_gradient[:, y_true] += x

            # Update weights
            self.w = (1 - eta * self.lambda_) * self.w - eta * loss_gradient

# the update step w = w − η · ∇(f)
# new w = w(1 - eta · λ) - eta · y · x after some rewriting


class LogisticRegressionMulticlass(LinearClassifier):
    """
    Multi-class logistic regression for assignment PA4, using One-vs-rest approach
    """

    def __init__(self, epochs=30, lambda_=1e-5):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.epochs = epochs
        self.lambda_ = lambda_

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """
        # If necessary, convert to dense format
        if not isinstance(X, np.ndarray):
            try: 
              X = X.toarray() # if list
            except: 
              X = [np.array(x) for x in X].toarray() # if list of lists

        encoder = OneHotEncoder()
        # our label matrix, one col per class:
        encoded_labels = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

        # Initialize the weight vector matrix to all zeros.
        n_features = X.shape[1]
        n_labels = len(list(set(Y)))
        self.w = np.zeros((n_features, n_labels))
        #self.w = np.random.normal(0, 0.01, size=(n_features, n_labels))

        # Logistic Regression Multiclass algorithm

        t = 0 
        xy_pairs = list(zip(X, encoded_labels))
        n_iter = X.shape[0] * self.epochs

        for i in range(n_iter):

            #select random pair from xy_pairs, get x and y
            (x, y_true) = random.choice(xy_pairs)
            t += 1
            eta = 1 / (self.lambda_ * t)

            # Compute the output scores for this instance, one score per label
            scores = x.dot(self.w)

            y_pred = np.argmax(scores) # class labels are same as indices

            # Compute the softmax probabilities
            softmax_probs = softmax(scores - scores[y_pred]) # subtract the best score from all scores

            # compute the loss gradient
            # np.outer() multiplies feature vector x with the softmax probability vector for probability scaling
            loss_gradient = np.outer(x, softmax_probs) - np.outer(x, y_true)

            # Update weights (same as before)
            self.w = (1 - eta * self.lambda_) * self.w - eta * loss_gradient


