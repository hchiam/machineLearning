# machineLearning:
Simple test code for machine learning in the Python programming language.

No need to install a ton of things to import.  Just read some commented code and get it running quickly to gather some intuitions.

You can have it even simpler and just run code in your browser without installing anything:  [here](http://hchiam.blogspot.ca/2016/08/machine-learning-very-basic-code.html).

![simple net](https://github.com/hchiam/machineLearning/blob/master/pictures/simpleNet.jpg "a simple neural network with two input neurons and one output neuron for the 'guess'")

* `neuralNet1.py`:  version 1 example of a very simplified neural network, using **sensitivity** parameter.
* `neuralNet2.py`:  version 2 example of a very simplified neural network, with **weighting** based on "responsibilities" of different **inputs**. (This one seems really fast but may be unstable or naive because it's basically using learning sensitivity = 1.)
* `neuralNet3.py`:  version 3 example of a very simplified neural network that **combines** version 1 and version 2, combining sensitivity parameter and "responsibilities" of different inputs.

![layered net](https://github.com/hchiam/machineLearning/blob/master/pictures/layeredNet.jpg "a layered neural network with two input neurons, two hidden neurons, and one output neuron for the 'guess'")

* `neuralNet4_Layered.py`:  version 4 example of a neural network that kinda combines version 2 and version 3, with learning error **sensitivity**, but also with a **hidden layer**.  It also has a transformed version of the **sigmoid** function that goes from -1 to 1.

![sim net](https://github.com/hchiam/machineLearning/blob/master/pictures/neuralNetwork2-2.jpg "a simple neural network with two inputs neurons and two output neurons to let the 'predator' move around")   ![simulation](https://github.com/hchiam/machineLearning/blob/master/pictures/simulationScreenshot.png "a screenshot of the simulation")

* `predatorSim1D.py` and `predatorSim2D.py`:  animated simulations of a "predator" learning to move towards a target.  Based on [my turtle code](https://github.com/hchiam/code7/blob/master/problem3.py) and also a mix of my neural nets above:  sensitivity parameter, weightings, inputs, but no hidden layer.  See it run [here](http://hchiam.blogspot.ca/2016/08/machine-learning-basic-simulator.html) or [here](https://trinket.io/python/2aa598ffb6).

# machine learning web app:

![webApp](https://github.com/hchiam/machineLearning/blob/master/pictures/LearnGesture.png "a web app that tries to detect a gesture made by the mouse anywhere on the page")

Under ["webApp"](https://github.com/hchiam/machineLearning/blob/master/webApp) folder:
* `gestures.html`:  the "structure" of the presentation of the web page.
* `gestures.js`:  the "brains" of the web page.  Tries to detect a mouse gesture when the mouse runs over the "pad".

You can try out the web app and see my sample code here: [https://codepen.io/hchiam/pen/rrwQRa](https://codepen.io/hchiam/pen/rrwQRa).

# Extra installation required but still pretty simple:

* `neuralNet_iamtrask.py`:  "11 lines of code" for a 2-layer neural net code from [http://iamtrask.github.io/2015/07/12/basic-python-network](http://iamtrask.github.io/2015/07/12/basic-python-network).  Requires NumPy installed to run.

## Google Developers:
The next few code samples are based on "Machine Learning Recipes with Josh Gordon" at:  [https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal), which is also listed in the [Google Developers YouTube channel](https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw).

* `googleExample.py`:  machine learning in 6 lines of code, from [https://youtu.be/cKxRvEZd3Mw](https://youtu.be/cKxRvEZd3Mw), listed under the [Google Developers](https://www.youtube.com/user/GoogleDevelopers) channel on YouTube.  Requires scikit-learn (sklearn) installed to run.  Decision tree classifier.  Supervised learning.

* `googleExample2.py`:  decision tree classifier applied to Fisher's Iris flower data set, from [https://www.youtube.com/watch?v=tNa99PG8hR8](https://www.youtube.com/watch?v=tNa99PG8hR8), listed under the [Google Developers](https://www.youtube.com/user/GoogleDevelopers) channel on YouTube.  Requires scikit-learn and NumPy installed to run.  You can also get a visualization (watch the video for how).

* `googleExample3.py`:  a higher-level take on the decision tree classifier in `googleExample2.py`.  Requires scikit-learn installed to run.

* `googleExample4.py`:  building your our classifier.  `randomGuessClassifier()` and `KNearestNeighbours_barebones()`.

## Sirajology "Learn Python for Data Science" Challenges:

1. [https://github.com/hchiam/gender_classification_challenge](https://github.com/hchiam/gender_classification_challenge)
