
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
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class,
                         self.negative_class])
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])


class Perceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # Perceptron algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):

                # Compute the output score for this instance.
                score = x.dot(self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    self.w += y*x


class SVC(LinearClassifier):
    """
    First try for implementing the SVC algorithm for assignment PA4 /FJ
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
        self.find_classes(Y)

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

            #if i%10000 == 0:
            #    print(f'Iteration {i}')
            
            #select random pair from xy_pairs, get x and y
            (x, y) = random.choice(xy_pairs)
            t += 1
            eta = 1 / (self.lambda_ * t)

            # Compute the output score for this instance
            score = x.dot(self.w)

            # If there was an error, update the weights
            if y*score < 1:
                self.w = (1 - eta * self.lambda_) * self.w + eta * y * x
            else:
                self.w = (1 - eta * self.lambda_) * self.w

# the update step w = w − η · ∇(f)
# new w = w(1 - eta · λ) - eta · y · x after some rewriting

class LogisticRegression(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
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
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # Logistic Regression algorithm

        t = 0 #used iteratively in eta calc
        xy_pairs = list(zip(X, Ye))
        n_iter = X.shape[0] * self.epochs
        
        for i in range(n_iter):

            #if i%10000 == 0:
            #    print(f'Iteration {i}')
            
            #select random pair from xy_pairs, get x and y
            (x, y) = random.choice(xy_pairs)
            t += 1
            eta = 1 / (self.lambda_ * t)

            # Compute the output score for this instance
            score = x.dot(self.w)

            # If there was an error, update the weights
            loss = y /(1+np.exp(y*self.w*x))* x
            self.w = (1 - eta * self.lambda_) * self.w + eta * loss
            # eta · y/(1+exp(y·(w·x))) · x

# new w = old w - step * gradient of obj func
# gradient of obj funct = λ · w + ∇(Loss)

# new w = w - eta (λ · w + ∇(Loss)) 
# = w - (eta · λ · w - eta · ∇(Loss))

# For Hinge loss: w = w - (eta · λ · w - eta · (-y · x))
# w = w - eta · λ · w + eta · y · x
# w = w(1 - eta · λ) + eta · y · x which is what it says in the paper

# ∇(Log Loss) = -y/(1+exp(yi·(w·xi))) · x
# For Log loss: w = w - (eta · λ · w - eta · (-y/(1+exp(y·(w·x))) · x)
# w = w - eta · λ · w - eta · -y/(1+exp(y·(w·x))) · x
# w = w(1 - eta · λ) + eta · y/(1+exp(y·(w·x))) · x

'''
For report: the logistic regression classifier seems to give slightly better results than
the SVC classifier, by about 1 percent-unit. However, no comparative testing was done so 
we would like to say that they are comparable in performance. Computationally, the logistic 
regression algorithm (weight updates) took longer than the SVC algorithm. 
'''


############# The following part is for the optional task. ###################

### Sparse and dense vectors don't collaborate very well in NumPy/SciPy.
### Here are two utility functions that help us carry out some vector
### operations that we'll need.

def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)


class SparsePerceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm,
    assuming that the input feature matrix X is sparse.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))

        for i in range(self.n_iter):
            for x, y in XY:

                # Compute the output score for this instance.
                # (This corresponds to score = x.dot(self.w) above.)
                score = sparse_dense_dot(x, self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    # (This corresponds to self.w += y*x above.)
                    add_sparse_to_dense(x, self.w, y)



