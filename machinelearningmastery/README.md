(Example that already works: https://github.com/hchiam/learning-keras/blob/master/mnist_mlp.py (gets data for you).)

# Based on Jason Brownlee's MachineLearningMastery.com

[http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](http://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

[http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

    # create a neural network using Keras
    # http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy
    
    seed = 7
    numpy.random.seed(seed)
    
    # 1) DATA ("LABELLED")
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # 2) NEURAL NETWORK
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    
    # 3) TRAIN
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=2)
    
    # 4) EVALUATE ACCURACY (+LOSS)
    scores = model.evaluate(X, Y)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    # now pretend we have new input data X to predict labels for:
    
    # 5) PREDICT
    predictions = model.predict(X)
    rounded = [round(x[0]) for x in predictions]
    print(rounded)