function mouseMovingOverPad(event) {
    showCoords(event);
    var learn = toggleLearn();
    learnGesture(learn);
    if (learn == false) {
        var gesture = detectGesture();
        showGesture(gesture);
    }
}

function clearDetections() {
    document.getElementById("coords").innerHTML = "";
    document.getElementById("gesture").innerHTML = "";
}

function showCoords(event) {
    var x = event.clientX;
    var y = event.clientY;
    var coords = "(" + x + "," + y + ")";
    document.getElementById("coords").innerHTML = coords;
}

function showGesture(gesture) {
    document.getElementById("gesture").innerHTML = "Gesture: " + gesture + "?";
}

function toggleCheckboxText() {
    if (document.getElementById("learn").checked == true) {
        document.getElementById("yesno").innerHTML = " Learn gesture (ON)";
    } else {
        document.getElementById("yesno").innerHTML = " Learn gesture (off)";
    }
}

function detectGesture() {
    var gesture = "";
    // goes here:  have ML algorithm try to categorize as gesture or not
    return gesture;
}

function toggleLearn() {
    var learn = false;
    if (document.getElementById("learn").checked == true) {
        learn = true;
    }
    return learn;
}

function learnGesture(learn) {
    if (learn == true) {
        // goes here:  show ML algorithm an example gesture
    }
}