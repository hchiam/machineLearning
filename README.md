# machineLearning
Simple test code for machine learning.

![simple net](https://github.com/hchiam/machineLearning/blob/master/simpleNet.jpg "a simple neural network with two input neurons and one output neuron for the 'guess'")

* `neuralNet1.py`:  version 1 example of a very simplified neural network, using **sensitivity** parameter.
* `neuralNet2.py`:  version 2 example of a very simplified neural network, with **weighting** based on "responsibilities" of different **inputs**. (This one seems really fast but may be unstable or naive because it's basically using learning sensitivity = 1.)
* `neuralNet3.py`:  version 3 example of a very simplified neural network that **combines** version 1 and version 2, combining sensitivity parameter and "responsibilities" of different inputs.

![layered net](https://github.com/hchiam/machineLearning/blob/master/layeredNet.jpg "a layered neural network with two input neurons, two hidden neurons, and one output neuron for the 'guess'")

* `neuralNet4_Layered.py`:  version 4 example of a neural network that kinda combines version 2 and version 3, with learning error **sensitivity**, but also with a **hidden layer**.  It also has a transformed version of the **sigmoid** function that goes from -1 to 1.
* `neuralNet_iamtrask.py`:  2-layer neural net code from: http://iamtrask.github.io/2015/07/12/basic-python-network.  Requires NumPy installed to run.