#Machine Learning Web App:

* `gestures.html`:  the "structure" of the presentation of the web page.
* `gestures.js`:  the "brains" of the web page. Tries to detect a mouse gesture when the mouse runs over the "pad".

Uses a simple time delay neural network [https://en.wikipedia.org/wiki/Time_delay_neural_network](https://en.wikipedia.org/wiki/Time_delay_neural_network).

Instead of running the html file locally, you can try out the web app live here: [https://codepen.io/hchiam/pen/rrwQRa](https://codepen.io/hchiam/pen/rrwQRa).

![webApp](https://github.com/hchiam/machineLearning/blob/master/pictures/LearnGesture.png "a web app that tries to detect a gesture when the mouse runs over the 'pad'")

#Main Data Flow Steps:

gestures.html
    onmouseover="mouseMoving(event);"
gestures.js
    mouseMoving(event)
    learnGesture(event)
    updateSynapsesWeights()
    detectGesture(event)