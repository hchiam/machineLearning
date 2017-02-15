# The MIT License (MIT)
# 
# Copyright (c) 2016 Howard Chiam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
# --------------

import math

# modified version of neuralNet2.py and neuralNet2.py, combined
# --> now with a hidden layer! (and sensitivity)
# --> and with sigmoid! (Though slower than neuralNet2.py for one set of inputs, I need sigmoid() in order to avoid instability of numbers going to infinity when I introduce new inputs.)

# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity
# h=hidden node

i1 = 0 # input 1
i2 = 1 # input 2

h1 = 0 # hidden node 1
h2 = 0 # hidden node 2

# weights:
w1h1 = -1    # input 1 to hidden node 1
w1h2 = -1     # input 1 to hidden node 2
w2h1 = -1     # input 2 to hidden node 1
w2h2 = -1    # input 2 to hidden node 2
wh1 = -1      # hidden node 1 to output guess
wh2 = -1     # hidden node 2 to output guess

a = 1 # the correct "answer"

s = 0.5 # sensitivity to error

def sigmoid(x): # this keeps x within -1 and 1
    return (1 / (1 + math.exp(-x)) -0.5)*2 # -0.5)*2 so it can reach from -1 to +1 (not just 0 to 1)

def train(numOfIters):
    
    # declare all as global here so can use values initialized above this function
    global i1, i2, h1, h2, w1h1, w1h2, w2h1, w2h2, wh1, wh2, a, s
    
    print 'i =', round(i1,3), round(i2,3)
    print 's =', s
    
    for iter in range(numOfIters): # train by going through iterations
        
        # calculate "guess" from input and weights:
        h1 = sigmoid(i1*w1h1 + i2*w2h1)
        h2 = sigmoid(i1*w1h2 + i2*w2h2)
        g = sigmoid(h1*wh1 + h2*wh2)
        
        e = (a-g) # error
        
        # calculate change in weights based on output error and input
        dwh1 = e*h1
        dwh2 = e*h2
        dw1h1 = dwh1*i1
        dw1h2 = dwh2*i1
        dw2h1 = dwh1*i2
        dw2h2 = dwh2*i2
        
        # update weights:
        wh1 += dwh1*s
        wh2 += dwh2*s
        w1h1 += dw1h1*s
        w1h2 += dw1h2*s
        w1h1 += dw1h1*s
        w2h2 += dw2h2*s
        
        # print out to let us see what's happening:    print 'e =', round(e,3)
        #print 'i =', round(i1,3), round(i2,3)
        #print 'w =', round(w1h1,3), round(w1h2,3), round(w2h1,3), round(w2h2,3), '(i --> h)'
        #print 'h =', round(h1,3), round(h2,3)
        #print 'w =', round(wh1,3), round(wh2,3), '(h --> g)'
        print 'g =', round(g,3)
        #print 'e =', round(e,3)
        #print 'd =', round(dwh1,3), round(dwh2,3), round(dw1h1,3), round(dw1h2,3), round(dw2h1,3), round(dw2h2,3)
        #   print
        
    print 'GUESS = ', g, '************************************' # output final "guess"
    # expected output "guess" should approximate the "answer" = 1

# what follows is like the "main" method that calls the above functions and variables:

print'________________________________________________'

train(5)

print'________________________________________________'

# test how fast it learns if I change the input (and keep the "answer" the same)
i1 = 1

train(5)

print'________________________________________________'

# sigmoid tests:
#print 'sigmoid(-1) =', sigmoid(-1)
#print 'sigmoid(1) =', sigmoid(1)
#print 'sigmoid(-2) =', sigmoid(-2)
#print 'sigmoid(2) =', sigmoid(2)
#print 'sigmoid(-3) =', sigmoid(-3)
#print 'sigmoid(3) =', sigmoid(3)
#print 'sigmoid(-10) =', sigmoid(-10)
#print 'sigmoid(10) =', sigmoid(10)
#print 'sigmoid(-25) =', sigmoid(-25)
#print 'sigmoid(25) =', sigmoid(25)
#
#print'________________________________________________'