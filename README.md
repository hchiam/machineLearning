(Just one of the things I'm learning. https://github.com/hchiam/learning)

(even more ML stuff: https://github.com/hchiam/learning-ml)

# machineLearning:

Simple test code for machine learning / neural networks / artificial intelligence (ML/NN/AI) in the Python programming language. And some live JavaScript examples too, like this one: https://codepen.io/hchiam/full/QGOyaE (for best results, open in Chrome).

No need to install a ton of things to import (more sophisticated code further down do need installations). Just read some commented code and get it running quickly to gather some intuitions.

You can have it even simpler and just run code in your browser without installing anything: [here](http://hchiam.blogspot.ca/2016/08/machine-learning-very-basic-code.html).

![simple net](https://github.com/hchiam/machineLearning/blob/main/pictures/simpleNet.jpg "a simple neural network with two input neurons and one output neuron for the 'guess'")

- `neuralNet1.py`: version 1 example of a very simplified neural network, using **sensitivity** parameter.
- `neuralNet2.py`: version 2 example of a very simplified neural network, with **weighting** based on "responsibilities" of different **inputs**. (This one seems really fast but may be unstable or naive because it's basically using learning sensitivity = 1.)
- `neuralNet3.py`: version 3 example of a very simplified neural network that **combines** version 1 and version 2, combining sensitivity parameter and "responsibilities" of different inputs.

![layered net](https://github.com/hchiam/machineLearning/blob/main/pictures/layeredNet.jpg "a layered neural network with two input neurons, two hidden neurons, and one output neuron for the 'guess'")

- `neuralNet4_Layered.py`: version 4 example of a neural network that kinda combines version 2 and version 3, with learning error **sensitivity**, but also with a **hidden layer**. It also has a transformed version of the **sigmoid** function that goes from -1 to 1.

![sim net](https://github.com/hchiam/machineLearning/blob/main/pictures/neuralNetwork2-2.jpg "a simple neural network with two inputs neurons and two output neurons to let the 'predator' move around") ![simulation](https://github.com/hchiam/machineLearning/blob/main/pictures/simulationScreenshot.png "a screenshot of the simulation")

- `predatorSim1D.py` and `predatorSim2D.py`: animated simulations of a "predator" learning to move towards a target. Based on [my turtle code](https://github.com/hchiam/code7/blob/master/problem3.py) and also a mix of my neural nets above: sensitivity parameter, weightings, inputs, but no hidden layer. See it run [here](http://hchiam.blogspot.ca/2016/08/machine-learning-basic-simulator.html) or [here](https://trinket.io/python/2aa598ffb6).

# machine learning web app:

You can try out the following web app live on CodePen: [https://codepen.io/hchiam/full/rrwQRa](https://codepen.io/hchiam/full/rrwQRa).

![webApp](https://github.com/hchiam/machineLearning/blob/main/pictures/LearnGesture.png "a web app that tries to detect a gesture made by the mouse anywhere on the page")

Under ["webApp_MachineLearning_Gesture"](https://github.com/hchiam/webApp_MachineLearning_Gesture) folder:

- `gestures.html`: the "structure" of the presentation of the web page.
- `gestures.js`: the "brains" of the web page. Tries to detect a mouse gesture when the mouse runs over the "pad".

![neurons flashing](https://github.com/hchiam/machineLearning/blob/main/pictures/circle.gif "neurons flashing")

# Markov Word Generator - create words with the same "feel":

https://github.com/hchiam/word_gen

# Notes and Code from Udacity course AI for Robotics:

https://github.com/hchiam/ai_for_robotics

# Genetic Algorithm - applied to one of my linguistics projects:

https://github.com/hchiam/cogLang-geneticAlgo

# Extra installation required but still pretty simple:

- `neuralNet_iamtrask.py`: "11 lines of code" for a 2-layer neural net code from [http://iamtrask.github.io/2015/07/12/basic-python-network](http://iamtrask.github.io/2015/07/12/basic-python-network). Requires NumPy installed to run.

## Google Developers:

The next few code samples are based on "Machine Learning Recipes with Josh Gordon" at: [https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal), which is also listed in the [Google Developers YouTube channel](https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw).

- `googleExample.py`: machine learning in 6 lines of code, from [https://youtu.be/cKxRvEZd3Mw](https://youtu.be/cKxRvEZd3Mw), listed under the [Google Developers](https://www.youtube.com/user/GoogleDevelopers) channel on YouTube. Requires scikit-learn (sklearn) installed to run. Decision tree classifier. Supervised learning.

- `googleExample2.py`: decision tree classifier applied to Fisher's Iris flower data set, from [https://www.youtube.com/watch?v=tNa99PG8hR8](https://www.youtube.com/watch?v=tNa99PG8hR8), listed under the [Google Developers](https://www.youtube.com/user/GoogleDevelopers) channel on YouTube. Requires scikit-learn and NumPy installed to run. You can also get a visualization (watch the video for how).

- `googleExample3.py`: a higher-level take on the decision tree classifier in `googleExample2.py`. Requires scikit-learn installed to run.

- `googleExample4.py`: building your our classifier. `randomGuessClassifier()` and `KNearestNeighbours_barebones()`.

- `googleExample5.py` and `googleExample6.md`: image classification examples.

## machinelearningmastery.com:

[https://github.com/hchiam/machineLearning/blob/main/machinelearningmastery](https://github.com/hchiam/machineLearning/blob/main/machinelearningmastery)

## Sirajology "Learn Python for Data Science" Challenges:

1. [https://github.com/hchiam/gender_classification_challenge](https://github.com/hchiam/gender_classification_challenge)

## Natural Evolution Strategies (NES) Example:

`nes.py`. See https://blog.openai.com/evolution-strategies/

## Keras

https://github.com/hchiam/learning-keras

## Synaptic.js

A JavaScript neural network library. My example codepen:

https://codepen.io/hchiam/pen/gWydZd?editors=1010

## ml5.js web-friendly machine learning, built on TensorFlow.js

https://codepen.io/hchiam/pen/LrJVPQ

## NLP with spaCy and textacy

https://github.com/hchiam/nlp_spacy_textacy

## Learn more with freeCodeCamp:

For example, here's a video I found helpful for understanding RNNs and LSTM: https://www.freecodecamp.org/learn/machine-learning-with-python/how-neural-networks-work/recurrent-neural-networks-rnn-and-long-short-term-memory-lstm

[LSTM: forget gate to forget irrelevant, input gate to remember relevant, and output gate to update new info](https://www.cloudskillsboost.google/course_sessions/6505024/video/363229)

Then later reading up on attention and Transformers makes more sense.

## Crash Course AI:

https://github.com/hchiam/crash-course-ai-labs

## AutoML:

https://github.com/hchiam/learning-automl

## Keep up to date:

https://www.youtube.com/@statquest - like this [clear explanation of ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) or of [transformers](https://www.youtube.com/watch?v=zxQyTK8quyY)

https://www.youtube.com/@TwoMinutePapers

https://www.youtube.com/@twimlai - hear about things like AI-GAs, Quality-Diversity algorithms, jailbreaking, filters, adversarial training, pre-training, and more.

## Links to more:

https://github.com/hchiam/learning-ml
