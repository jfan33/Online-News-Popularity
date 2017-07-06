# Final Project
# By Jessie Fan

# Objective: Design a program to distinguish classes in Online News Popularity
# Design: Classification task
# Performance measure: Accuracy

from __future__ import division, print_function, unicode_literals

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, FastICA, NMF

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import time

# Load data, by specifying "path" it will be easier for modification
# I've tried to load data directly through url, however it's very strenuous to unzip original file...
path = "/Users/jessiefan/Desktop/"
news = pd.read_csv(path+"OnlineNewsPopularity.csv", thousands=',')

# Learn data (data structure)
# print(news.info())
# print(news.columns)

# Data Pre-processing
# using 1400 as threshold for "share"
popular = news[" shares"] >= 1400
unpopular = news[" shares"] < 1400

# update original data set based on threshold
news.loc[popular, ' shares'] = 1
news.loc[unpopular, ' shares'] = 0

# Data Modeling
# Features selection
features = news.columns[2:60]
X = StandardScaler().fit_transform(news[features])
y = news[' shares'].ravel()

'''
# Annotated this section because none of the techniques helped improving accuracy
# Dimension reduction: PCA
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
X_new = pca.transform(X)

# Kernel PCA, Sparse PCA
# Computationally inefficient
kpca = KernelPCA()
X_kpca = kpca.fit_transform(X)


X_spca = SparsePCA().fit_transform(X)


# ICA
# Didn't improve accuracy either
X_ica = FastICA().fit_transform(X)

# NMF
# not feasible as there are negative values in data passed to NMF
X_nmf = NMF().fit_transform(X)
'''

# Split data using train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(feature, target, weight):
    scores = np.dot(feature, weight)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(feature, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((feature.shape[0], 1))
        feature = np.hstack((intercept, feature))

    weight = np.zeros(feature.shape[1])

    for step in range(num_steps):
        scores = np.dot(feature, weight)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(feature.T, output_error_signal)
        weight += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print(log_likelihood(feature, target, weight))
    return weight

features = np.vstack(X.astype(np.float32))
labels = np.hstack(y)

weight = logistic_regression(X_train, y_train, num_steps=30000, learning_rate=5e-5, add_intercept=True)
data_with_intercept = np.hstack((np.ones((features.shape[0], 1)), features))
final_scores = np.dot(data_with_intercept, weight)
pred = np.round(sigmoid(final_scores))

print('Implemented Logistic Regression: {0}'.format((pred == labels).sum().astype(float) / len(pred)), '\n')

# Create a list for names of all classifiers
# Both QDA and LDA require the assumption that the features are not correlated.
# Therefore not included in actual learning, although accuracy is high.
names = [
    "Naive Bayes",
    "Logistic",
    "Gradient Descent",
    "Perceptron",
    "Neural Network",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Gradient Boosting",
    "Bagging",
    # "QDA",
    # "LDA",
    "SVM-RBF"
]

# Create a list for all classifiers
# parameter setting shown in list are the combinations where I found optimum accuracy at

classifiers = [
    BernoulliNB(binarize=-0.05),
    LogisticRegression(),
    SGDClassifier(),
    Perceptron(penalty='l1', alpha=0.00011),
    MLPClassifier(activation='identity'),
    DecisionTreeClassifier(criterion='gini', splitter='random'),
    RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False),
    AdaBoostClassifier(n_estimators=50, learning_rate=1.1),
    GradientBoostingClassifier(loss='deviance', learning_rate=0.18, n_estimators=150),
    BaggingClassifier(n_estimators=50),
    # QuadraticDiscriminantAnalysis(),
    # LinearDiscriminantAnalysis(),
    SVC(kernel='rbf'),
]

# The following for loop will iterate through lists of names and classifiers

for name, clf in zip(names, classifiers):
    start_time = time.time()
    clf.fit(X_train, y_train)
    predicted_IV = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    end_time = time.time()
    print(name, ':', format(score, '.3f'), '|time elapsed:', format((end_time - start_time), '.2f'), '\n')

# This design will automate learning process
# It enables user to modify the project easily
