var sampleTimer;

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
    var x = getCoords(event)[0];
    var y = getCoords(event)[1];
    var coords = "(" + x + "," + y + ")";
    document.getElementById("coords").innerHTML = coords;
}

function getCoords(event) {
    var x = event.clientX;
    var y = event.clientY;
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
        // goes here:  show ML algorithm an example gesture
        sampleTimer = setInterval(getSamples(event), 1000/10); // 10 per second
    }
}

function getSamples(event) {
    getCoords(event);
    // more stuff here
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