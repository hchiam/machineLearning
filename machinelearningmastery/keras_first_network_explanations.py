# create a neural network using Keras
# reference: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential # sequential model (sequence of layers)
from keras.layers import Dense # "Dense" class of layers = fully-connected layers (FCL)
import numpy

# set random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# get DATA from file
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split data into INPUT (X) and OUTPUT/LABELS (Y) variables
X = dataset[:,0:8] # all rows, columns 1 to 7 only
Y = dataset[:,8] # all rows, column 8 only

# create MODEL
model = Sequential() # sequential model from keras

# add LAYERS to model:
# (12, 8, 1 neurons in "dense"/fully-connected layers):
model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) # add layer 1: FCL with 12 neurons but 8 input variables
model.add(Dense(8, init='uniform', activation='relu')) # add layer 2: use ReLU for better performance than sigmoid
model.add(Dense(1, init='uniform', activation='sigmoid')) # add layer 3: use sigmoid for output from 0 to 1 and easy classifying

# set model TRAINING 'REGIMEN':
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# (loss = how set synapse weights, optimizer = how search through synapse weights, metrics = extra info to track)
# 'adam' = an efficient gradient descent algorithm
# 'accuracy' metric --> we'll use as classification accuracy

# TRAIN / fit model on the data: (call the model's fit() function):
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=2)
# epoch = iterations (set to limit number of trainings)
# batch size = number of samples to evaluate before update neural network synapse weights
# 'verbose' = optional ID for how much progress info to print back to user during model training iterations

# EVALUATE model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# now pretend we have new input data X to predict labels for:

# get LABEL PREDICTIONS
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]

# print out LABELS for input, in order
print(rounded)
