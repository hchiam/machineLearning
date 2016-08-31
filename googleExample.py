from sklearn import tree # Watch source video for explanation:  https://youtu.be/cKxRvEZd3Mw in the Google Developers channel
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[160, 0]]) # 0 = apple ; 1 = orange