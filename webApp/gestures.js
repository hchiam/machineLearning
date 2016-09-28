var sampleTimer;
var padWidth = 100;
var padHeight = 100;
var shiftx = -9;
var shifty = -9;
var neuralNet = create3DMatrix(50,3,3);

function mouseMovingOverPad(event) { // I'd recommend you read the code starting from here
    showCoords(event);
    var learn = toggleLearn();
    learnGesture(learn, event);
    if (learn == false) {
        var gesture = detectGesture();
        showGesture(gesture);
    }
}

function showCoords(event) {
    var x = getCoords(event)[0] + shiftx;
    var y = getCoords(event)[1] + shifty;
    var coords = "(" + x + "," + y + ")";
    document.getElementById("coords").innerHTML = coords;
}

function getCoords(event) {
    var x = event.clientX + shiftx;
    var y = event.clientY + shifty;
    return [x,y];
}

function toggleLearn() {
    var learn = false;
    if (document.getElementById("learn").checked == true) {
        learn = true;
    }
    return learn;
}

function learnGesture(learn, event) {
    if (learn == true) {
        sampleTimer = setInterval(updateNeuralNetwork(event), 1000*2); // 10 per second = 1000/10 ; 1 per 2 seconds = 1000*2
    }
}

function updateNeuralNetwork(event) {
    shiftSamples(event); // show ML algorithm an example gesture
    setWeights(event); // have ML algorithm set neuron synapse weights
}

function shiftSamples(event) {
    var inputSectionMatrix = getPadSection(event);
    var numberOfSnapshots = neuralNet.length;
    for (i = numberOfSnapshots-1; i > 0; i--) {
        neuralNet[i] = neuralNet[i-1];
    }
    neuralNet[0] = inputSectionMatrix; // example:  [[0,0,0],[0,0,0],[1,0,0]]
    // debug output:
    document.getElementById("section").innerHTML = neuralNet;
}

function create3DMatrix(snapshots,rows,columns) {
    var x = snapshots;
    var y = rows;
    var z = columns;
    var matrix = new Array(x);
    for (i = 0; i < x; i++) {
        matrix[i] = new Array(y);
        for (j = 0; j < y; j++) {
            matrix[i][j] = new Array(z);
            for(k = 0; k < z; k++) {
                matrix[i][j][k] = 0;
            }
        }
    }
    return matrix;
}

function getPadSection(event) {
    var sectionMatrix = [
                   [0,0,0],
                   [0,0,0],
                   [0,0,0]
                   ];
    var sectionx, sectiony;
    var coords = getCoords(event); // [x,y]
    var x = coords[0];
    var y = coords[1];
    // get section's x coordinate
    if (x <= padWidth / 3) {
        sectionx = 0;
    } else if (x > padWidth / 3 && x <= padWidth * 2/3) {
        sectionx = 1;
    } else if (x > padWidth * 2/3) {
        sectionx = 2;
    }
    // get section's y coordinate
    if (y <= padHeight / 3) {
        sectiony = 0;
    } else if (y > padHeight / 3 && y <= padHeight * 2/3) {
        sectiony = 1;
    } else if (y > padHeight * 2/3) {
        sectiony = 2;
    }
    // detect the section of the pad
    sectionMatrix[sectionx][sectiony] = 1;
    // debug output:
    document.getElementById("s").innerHTML = ['section=['+sectionx+','+sectiony+'] matrix='+sectionMatrix+' coords=('+x+','+y+')'];
    return sectionMatrix;
}

function setWeights() {
    
    
}

function detectGesture() {
    var gesture = "";
    // goes here:  have ML algorithm try to categorize as gesture or not
    return gesture;
}

function showGesture(gesture) {
    document.getElementById("gesture").innerHTML = "Gesture: " + gesture + "?";
}

/*
 /*
 /* other functions based on other non-synchronous actions:
 /*
 */

function toggleCheckboxText() {
    if (document.getElementById("learn").checked == true) {
        document.getElementById("yesno").innerHTML = " Learn gesture (ON)";
    } else {
        document.getElementById("yesno").innerHTML = " Learn gesture (off)";
    }
}

function clearDetections() { // "end" with this function
    document.getElementById("coords").innerHTML = "";
    //document.getElementById("gesture").innerHTML = "";
    document.getElementById("learn").checked = false; // auto stop learning when mouse leaves "pad"
    toggleCheckboxText(); // make checkbox text match checkbox state
}