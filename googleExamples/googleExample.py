from sklearn import tree # Watch source video for explanation:  https://youtu.be/cKxRvEZd3Mw in the Google Developers channel
features = [[140, 1], [130, 1], [150, 0], [170, 0]] # weights and bumpiness for each example
labels = [0, 0, 1, 1] # labels for each example
clf = tree.DecisionTreeClassifier() # create a decision tree classifier
clf = clf.fit(features, labels) # fit features and labels; "find patterns in the data"
print clf.predict([[160, 0]]) # 0 = apple ; 1 = orange