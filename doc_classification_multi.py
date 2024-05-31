
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from aml_perceptron_multi import LogisticRegressionMulticlass

# This function reads the corpus, returns a list of documents, and a list
# of their corresponding polarity labels. 
def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            y, _, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y


if __name__ == '__main__':
    
    # Read all the documents.
    X, Y = read_data('data/all_sentiment_shuffled.txt')
    
    # Split into training and test parts.
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

    # Set up the preprocessing steps and the classifier.
    pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        #OneVsOneClassifier(SVC())
        #OneVsOneClassifier(LogisticRegression())
        #OneVsRestClassifier(SVC())
        LogisticRegressionMulticlass()
    )

    # encode labels to numbers 
    # (can't use encoded Ytrain since the pipeline requires 1D array)
    encoder = OneHotEncoder()
    encoder.fit_transform(np.array(Ytrain).reshape(-1, 1)).toarray()
    encoded_Ytest = encoder.transform(np.array(Ytest).reshape(-1, 1)).toarray()
    encoded_Ytest = encoded_Ytest.argmax(axis=1)

    # Train the classifier.
    t0 = time.time()
    pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    print('Training time: {:.2f} sec.'.format(t1-t0))

    # Evaluate on the test set.
    Yguess = pipeline.predict(Xtest)
    print('Accuracy: {:.4f}.'.format(accuracy_score(encoded_Ytest, Yguess)))


