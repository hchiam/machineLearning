# High level code for an image classifier of a directory of images using tensorflow for poets, which takes care of setting up an training a deep learning neural network.

# The name of the game is diversity and quantity of data.



# make directories for each type of flower
# get images from code lab

# train classifier with deep learning from tensorflow for poets

from sklearn import metrics, cross_validation
import tensorflow as tf
from tensorflow.contrib import learn



def main(usused_argv):
    # Load dataset.
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

    # Fit and predict.
    classifier.fit(x_train, y_train, setps=200)
    score = metrics.accuracy_score(y_test, classifier.predict(x_test))
    print('Accuracy: {0:f}'.format(score))

