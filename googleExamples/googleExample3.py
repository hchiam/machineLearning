# for episode 4 from https://www.youtube.com/watch?v=84gqSbLcBFE

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
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# train classifier
my_classifier.fit(X_train, y_train)

# test classifier
predictions = my_classifier.predict(X_test)
print '\npredictions =', predictions

# calculate accuracy of predictions against the true labels in y_test
from sklearn.metrics import accuracy_score
print '\naccuracy =', accuracy_score(y_test, predictions)

print "\a"