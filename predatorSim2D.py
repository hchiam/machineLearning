# Here are some definitions of the short forms:
# i=input, w=weight, o=output, e=error, s=sensitivity
# x,y = position of predator; xT,yT = position of target (prey)

import math
import time # this will enable drawing slower "frames"
import turtle # this will enable simple drawing

# global variables (define before the functions that use them)
global x
global y
global xT
global yT
global o1,o2,e # these are global so that printout printWhatsGoingOn() can be used

def drawDotHere(x,y,colour):
    turtle.penup()
    turtle.goto(x,y)
    turtle.pendown()
    turtle.pencolor(colour)
    turtle.dot(50)

def detectTarget():
    threshold = 1
    if abs(xT-x) > threshold:
        if x < xT:
            leftEye, rightEye = 0, 1
        elif x > xT:
            leftEye, rightEye = 1, 0
    else:
        leftEye, rightEye = 0, 0
    return leftEye, rightEye

def sigmoid(x): # this keeps x within -1 and 1
    return (1 / (1 + math.exp(-x)) -0.5)*2 # -0.5)*2 so it can reach from -1 to +1 (not just 0 to 1)

def r(x): # convenience rounding function
    return round(x,2)

def printWhatsGoingOn():
    print("output1= "+str(r(o1))+"\toutput2= "+str(r(o2))+"\terror= "+str(r(e))+"\tw: "+str(r(w11))+"  "+str(r(w12))+"  "+str(r(w21))+"  "+str(r(w22))) # print out to let us see what's happening

#_________________________

#HERE IS WHERE THE "MAIN FUNCTION" BEGINS:
#_________________________


print("________________________")
print("Simulation starting.")
print("________________________")


# set up drawing:
turtle.hideturtle() # hide the turtle drawer
turtle.speed(0) # this makes it go at full speed per "frame"
turtle.bgcolor("light grey")


# initialize positions:
x,y,xT,yT = -100,0,100,0
drawDotHere(xT,yT,"blue")
drawDotHere(x,y,"red")
xORIGINAL,yORIGINAL,xTORIGINAL,yTORIGINAL = x,y,xT,yT


# initialize variables:
i1, i2 = detectTarget() # input values
w11 = -1 # weight i1 to o1
w12 = 1 # weight i1 to o2
w21 = 0 # weight i2 to o1
w22 = -1 # weight i2 to o2
o1 = i1*w11 + i2*w21 # calculate output from input and weights
o2 = i1*w12 + i2*w22 # calculate output from input and weights
o1 = sigmoid(o1) # keep output value within allowed range
o2 = sigmoid(o2) # keep output value within allowed range
s = 0.1 # sensitivity to error
speed = 20 # speed of motion of the "organism"


# train by going through iterations:
for iter in range(30):
    
    time.sleep(0.6) # delay each "frame" (also avoid giving people headaches with the flashing colours)
    
    # get input values:
    i1, i2 = detectTarget()
    
    # get error:
    e = -i1 +i2
    
    # get changes in weights based on error and input values:
    dw11 = e*i1 # calculate change in weight 1-1 based on output error and input 1
    dw12 = e*i1 # calculate change in weight 1-2 based on output error and input 1
    dw21 = e*i2 # calculate change in weight 2-1 based on output error and input 2
    dw22 = e*i2 # calculate change in weight 2-2 based on output error and input 2
    
    # adjust weights:
    w11 += dw11*s
    w12 += dw12*s
    w21 += dw21*s
    w22 += dw22*s
    
    # get output values:
    o1 = i1*w11 + i2*w21 # calculate output from input and weights
    o2 = i1*w12 + i2*w22 # calculate output from input and weights
    o1 = sigmoid(o1) # keep output value within allowed range
    o2 = sigmoid(o2) # keep output value within allowed range
    
    # adjust position:  (move the "muscles" of the "organism")
    x += o1*speed +o2*speed
    
    # draw new "frame" of "animation":
    turtle.clear()
    if e != 0:
        drawDotHere(xTORIGINAL,yTORIGINAL,"light blue")
    drawDotHere(xORIGINAL,yORIGINAL,"pink")
    if e != 0:
        drawDotHere(xT,yT,"blue")
    drawDotHere(x,y,"red")
    
    # print out values so user can see changes in error, etc.:
    printWhatsGoingOn()


# print out final values:
printWhatsGoingOn()


# let user they can close the drawing/"animation":
print("________________________")
print("Simulation done.")
print("________________________")
print("Click the drawing window to close it.")
print("________________________")
turtle.exitonclick()