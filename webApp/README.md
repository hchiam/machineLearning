#Machine Learning Web App:

* `gestures.html`:  the "structure" of the presentation of the web page.
* `gestures.js`:  the "brains" of the web page. Tries to detect a mouse gesture when the mouse runs over the "pad".

Uses a simple time delay neural network [https://en.wikipedia.org/wiki/Time_delay_neural_network](https://en.wikipedia.org/wiki/Time_delay_neural_network).

Instead of running the html file locally, you can try out the web app live here: [https://codepen.io/hchiam/pen/rrwQRa](https://codepen.io/hchiam/pen/rrwQRa).

![webApp](https://github.com/hchiam/machineLearning/blob/master/pictures/LearnGesture.png "a web app that tries to detect a gesture when the mouse runs over the 'pad'")

#Main Data Flow Steps:

* gestures.html

    1) onmouseover="mouseMoving(event);"

* gestures.js

    2) mouseMoving(event)

    3) learnGesture(event)

    4) updateSynapsesWeights()

    5) detectGesture(event)

#Example Gesture:

Making two quick clockwise circles with the mouse.  The following synapse weights and parameters make this happen.  The neural net can distinguish the motion from simple mouse cursor swipes, and even discriminate clockwise from counterclockise.  Small motions are automatically "filtered out" because of the thresholdMovementSize being > 0.  Training takes a while with confidenceThreshold = 90, but also helps "filter out" most false positives.

## Parameters/Version that worked for "two quick clockwise circles with the mouse":

For gestures.js, with "confidence > 90, movement dx and dy both > 5 then 0 detection "; commit 8eddc91cbcdf7d5be357abafe91f12a1efb866a9

```
var confidenceThreshold = 90;
var thresholdMovementSize = 5;
if (Math.abs(dx) < thresholdMovementSize && Math.abs(dy) < thresholdMovementSize) {
    directionMatrix[directionx][directiony] = 0;
}
```

##Example Synapse Weights for "two quick clockwise circles with the mouse":

wts=1,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0.99,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1
