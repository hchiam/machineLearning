# for episode 5 from https://www.youtube.com/watch?v=AoeEHqVSNOw

import random

class randomGuessClassifier():

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions



from scipy.spatial import distance

def eucDist(a,b):
    return distance.euclidean(a,b)

class KNearestNeighbours():
    
    def fit(self, X_train, y_train): # "memorize" the x data and corresponding y labels
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test): # X_test has different features we treat as dimensions --> label each row of features (i.e. data point) with the same label as the nearest neighbour(s)
        predictions = []
        for row in X_test:
            label = self.closest(row) # note:  default k = 1 nearest neighbour's label to get for each row (i.e. data point)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = eucDist(row, self.X_train[0]) # dist from test to first train point
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = eucDist(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index] # return label of closest training example



# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# f(x) = y ; features(x) = label
X = iris.data
y = iris.target

# split into test and train data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# create classifier
#my_classifier = randomGuessClassifier()
my_classifier = KNearestNeighbours()

# train classifier
my_classifier.fit(X_train, y_train)

# test classifier
predictions = my_classifier.predict(X_test)
print '\npredictions =', predictions

# calculate accuracy of predictions against the true labels in y_test
from sklearn.metrics import accuracy_score
print '\naccuracy =', accuracy_score(y_test, predictions)

print "\a"



# K nearest neighbours:

# Simple but slow.
# Hard to represent how some features are more informative.