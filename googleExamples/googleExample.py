# Watch source video for explanation: https://youtu.be/cKxRvEZd3Mw in the Google Developers channel

# IMPORT TREE module from scikit-learn
from sklearn import tree

# get DATA set of features corresponding to each label
features = [[140, 1], [130, 1], [150, 0], [170, 0]] # weights and bumpiness for each example

# get LABELS (with indices corresponding to the respective features indices)
labels = [0, 0, 1, 1] # labels for each example

# create a decision tree CLASSIFIER
clf = tree.DecisionTreeClassifier()

# TRAIN the classifier
clf = clf.fit(features, labels) # fit features and labels; "find patterns in the data"

# print OUTPUT
print(clf.predict([[160, 0]])) # 0 = apple ; 1 = orange
