# Decision tree applied to Fisher's Iris flower data set
# Code based on tutorial https://www.youtube.com/watch?v=tNa99PG8hR8 from Google Developers YouTube channel.

# notes:
# 1) data (know how to work with the data set)
# 2) train (
# 3) test (for "never seen before" data)

# This example code works with the Fisher's Iris flower data set, as seen in https://en.wikipedia.org/wiki/Iris_flower_data_set

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
#print iris.feature_names
#print iris.target_names
#print iris.data[0]

test_idx = [0,50,100] # will "remove" these indices from the training data set (one for each iris type)

# set the training labels and data (on the remaining data)
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# set the testing labels and data (on the examples "removed")
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print 'target labels =\n ', test_target # this are the target labels for our algorithm:  it should show flowers of each type [0, 1, 2]
print 'tree-predicted labels =\n ', clf.predict(test_data) # features --(tree)--> our tree's predicted labels

# insert graphviz code here to see decision tree visualization generated as a pdf file
# this is cool!  it automatically creates a kind of taxonomy!

print 'data = ', test_data[0]
print 'label = ', test_target[0]

print 'feature names =\n ', iris.feature_names
print 'target names =\n ', iris.target_names

# Closing point:
# Every question the decision tree asks must be about one of the features you choose.
# Therefore, the better the features, the better the decision tree you can build.
# What makes a good feature?
