import numpy as np

f = open("data.csv")
f.readline() # skip the first row
data = np.loadtxt(f)

features = data[:,[0,1,2,3]]
labels = data[:,4]

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0, 1, 2]]))