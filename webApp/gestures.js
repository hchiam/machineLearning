// I'd recommend you read the code starting from "function mouseMoving(event)" to get the main idea

document.addEventListener("mousemove", mouseMoving); // detect mouse position anywhere on page

var sampleTimer;
var padWidth = window.innerWidth;
var padHeight = window.innerHeight;
var shiftx = -9;
var shifty = -9;
var snapshots = 20;
var rows = 3;
var columns = 3;
var neuralNet = create3DMatrix(snapshots,rows,columns);
var xNN = columns;
var yNN = rows;
var zNN = snapshots;
var numOfWts = xNN * yNN * zNN;
var wts = create3DMatrix(snapshots,rows,columns); // so guarantee same size
var testInputMatrix = create3DMatrix(snapshots,rows,columns); // so guarantee same size

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

function create2DMatrix(rows,columns) {
    var y = rows;
    var z = columns;
    var a = "";
    var matrix = new Array(y);
    for (j = 0; j < y; j++) {
        matrix[j] = new Array(z);
        for(k = 0; k < z; k++) {
            matrix[j][k] = 0;
        }
    }
    return matrix;
}

function mouseMoving(event) { // I'd recommend you read the code starting from here
    showDirectionVector(event); // showCoords(event);
    var learn = toggleLearn();
    if (learn === true) {
        learnGesture(event);
    } else if (learn === false) {
        // stop learning
        clearTimeout(sampleTimer);
        updateSynapsesWeights(); // have ML algorithm set neuron synapse weights
        // start detecting
        var gesture = detectGesture(event);
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

function showDirectionVector(event) {
    var x = getDirectionVector(event)[0];
    var y = getDirectionVector(event)[1];
    var vector = "(" + x + "," + y + ")";
    document.getElementById("coords").innerHTML = vector;
}

function getDirectionVector(event) {
    var x = event.movementX;
    var y = event.movementY;
    return [x,y];
}

function toggleLearn() {
    var learn = false;
    if (document.getElementById("learn").checked === true) {
        learn = true;
    }
    return learn;
}

function learnGesture(event) {
    sampleTimer = setInterval(updateNeuralNetwork(event, neuralNet), 1000*2); // 1 per 2 seconds = 1000*2 ; 10 per second = 1000/10
}

function updateNeuralNetwork(event, matrix) {
    shiftSamples(event, matrix); // show ML algorithm an example gesture
    //updateSynapsesWeights(); // have ML algorithm set neuron synapse weights
}

function shiftSamples(event, matrix) {
    var inputSectionMatrix = getVelocityDirection(event); // getPadSection(event);
    var numberOfSnapshots = matrix.length;
    for (i = numberOfSnapshots-1; i > 0; i--) {
        matrix[i] = matrix[i-1];
    }
    matrix[0] = inputSectionMatrix; // example:  [[0,0,0],[0,0,0],[1,0,0]]
    // debug output:
    document.getElementById("section").innerHTML = matrix;
}

function getPadSection(event) {
    var sectionMatrix = create2DMatrix(3,3);
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

function getVelocityDirection(event) {
    var directionMatrix = create2DMatrix(3,3);
    var directionx, directiony;
    var vector = getDirectionVector(event); // [x,y]
    var dx = vector[0];
    var dy = vector[1];
    var slope = dy/dx;
    // get section's x coordinate
    if (dx < 0) {
        if (slope > 2.414) {
            directionx = 1;
            directiony = 0;
        } else if (slope <= 2.414 && slope > 0.414) {
            directionx = 0;
            directiony = 0;
        } else if (slope <= 0.414 && slope > -0.414) {
            directionx = 0;
            directiony = 1;
        } else if (slope <= -0.414 && slope > -2.414) {
            directionx = 0;
            directiony = 2;
        } else if (slope <= -2.414) {
            directionx = 1;
            directiony = 2;
        }
    } else if (dx === 0) {
        if (slope >= 2.414 || slope <= -2.414) {
            if (dy > 0) {
                directionx = 1;
                directiony = 2;
            } else if (dy === 0) {
                directionx = 1;
                directiony = 1;
            } else if (dy < 0) {
                directionx = 1;
                directiony = 0;
            }
        } else if (slope < 2.414) {
            if (dy > 0) {
                directionx = 2;
                directiony = 2;
            } else if (dy === 0) {
                directionx = 1;
                directiony = 1;
            } else if (dy < 0) {
                directionx = 0;
                directiony = 0;
            }
        } else if (slope > -2.414) {
            if (dy > 0) {
                directionx = 0;
                directiony = 2;
            } else if (dy === 0) {
                directionx = 1;
                directiony = 1;
            } else if (dy < 0) {
                directionx = 2;
                directiony = 0;
            }
        }
    } else if (dx > 0) {
        if (slope > 2.414) {
            directionx = 1;
            directiony = 2;
        } else if (slope <= 2.414 && slope > 0.414) {
            directionx = 0;
            directiony = 2;
        } else if (slope <= 0.414 && slope > -0.414) {
            directionx = 0;
            directiony = 1;
        } else if (slope <= -0.414 && slope > -2.414) {
            directionx = 0;
            directiony = 0;
        } else if (slope <= -2.414) {
            directionx = 1;
            directiony = 0;
        }
    }
    //// get section's y coordinate
    //if (dy < 0) {
    //    directiony = 0;
    //} else if (dy === 0) {
    //    directiony = 1;
    //} else if (dy > 0) {
    //    directiony = 2;
    //}
    // detect the section of the pad
    directionMatrix[directionx][directiony] = 1;
    // debug output:
    document.getElementById("s").innerHTML = ['section=['+directionx+','+directiony+'] matrix='+directionMatrix+' vector=('+dx+','+dy+')'];
    return directionMatrix;
}

function updateSynapsesWeights() {
    // numOfWts = number of weights, already defined at top
    // wts = matrix of weights, already defined at top
    var x = zNN;
    var y = yNN;
    var z = xNN;
    var sensitivity = 0.1;
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            for(k = 0; k < z; k++) {
                wts[i][j][k] += neuralNet[i][j][k] * sensitivity;
                wts[i][j][k] = round( sigmoid( wts[i][j][k] ), 2 );
            }
        }
    }
    document.getElementById("wts").innerHTML = ['wts='+wts];
}

function round(x,digits) {
    return Math.round(x * Math.pow(10,digits)) / Math.pow(10,digits);
}

function sigmoid(x) { // to keep number range within 0 to 1
    return (1 / (1 + Math.exp(-x*6)) -0.5)*2;
    // "-0.5)*2" because want input=0 to give output=0
    // "-x*6" because want to compress plot to have input ranging from 0 to 1 (and not 0 to 6)
}

function detectGesture(event) {
    var gesture = "";
    var outputValue = 0;
    var x = zNN;
    var y = yNN;
    var z = xNN;
    // track the input gesture
    trackGesture(event);
    // have ML algorithm try to categorize as gesture or not
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            for(k = 0; k < z; k++) {
                weight = wts[i][j][k];
                input = testInputMatrix[i][j][k];
                outputValue += weight * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
            }
        }
    }
    // get final, rounded percent output value
    outputValue = round(outputValue,2); // round to 2 decimal places
    outputValue = outputValue*100; // get percentage
    // debug output
    document.getElementById("confidence").innerHTML = "confidence="+outputValue+"%";
    document.getElementById("meter").value = outputValue/100;
    if (outputValue > 80) {
        gesture = "DETECTED!";
        document.getElementById("pad").style.backgroundColor = "green";
    } else {
        gesture = "?";
        document.getElementById("pad").style.backgroundColor = "red";
    }
    return gesture;
}

function trackGesture(event) {
    sampleTimer = setInterval(updateNeuralNetwork(event, testInputMatrix), 1000*2); // 1 per 2 seconds = 1000*2 ; 10 per second = 1000/10
}

function showGesture(gesture) {
    document.getElementById("gesture").innerHTML = "Gesture:  " + gesture;
}

/*
 /*
 /* other functions based on other non-synchronous actions:
 /*
 */

function toggleCheckboxText() {
    if (document.getElementById("learn").checked === true) {
        document.getElementById("yesno").innerHTML = "LEARNING gesture (press spacebar to stop)";
        document.getElementById("pad").style.backgroundColor = "blue";
    } else {
        document.getElementById("yesno").innerHTML = "NOT learning gesture (press spacebar to start)";
        document.getElementById("pad").style.backgroundColor = "red";
    }
}

function clearDetections() { // "end" with this function
    document.getElementById("coords").innerHTML = "";
    //document.getElementById("gesture").innerHTML = "";
    document.getElementById("learn").checked = false; // auto stop learning when mouse leaves detection area
    toggleCheckboxText(); // make checkbox text match checkbox state
}

$(document).ready(function() {
	document.getElementById("attrib").innerHTML = "Written and coded by <a href='http://codepen.io/hchiam'>hchiam</a>";
});