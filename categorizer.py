import numpy as np
from sklearn.neural_network import MLPClassifier

def get_data(file_name):
    f = open(file_name)
    f.readline() # skip the first row
    data = np.loadtxt(f)
    return data

def get_features(data):
    # expect label to be last column
    return data[:,:-1]

def get_labels(data):
    # expect label to be last column
    return data[:,-1]

def create_classifier(features, labels):
    clf = MLPClassifier()
    clf = clf.fit(features, labels)
    return clf

def run():
    data_train = get_data('data.csv')
    features = get_features(data_train)
    labels = get_labels(data_train)
    categorizer = create_classifier(features, labels)
    data_test = get_data('data-unlabeled.csv')
    for example in data_test:
        print(categorizer.predict([example]))

run()
