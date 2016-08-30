# My attempt to create a super simple (if over-simplified) version of a VAE (Variational Auto-Encoder).

# See http://arxiv.org/pdf/1606.05908v2.pdf for Carl Doersch's tutorial on VAE's.

# I personally recommend looking at Figure 4's right side (page 10) and Figure 5 (page 11) for the intuition behind a VAE:
# Blue blocks = to be optimized (minimize difference of output-input and minimize the "difference" of 2 probabilities).
# Red blocks = sampling from normalized distributions with mean & variance in brackets:  N(mean, variance).
# Also see explanation in Section 2.4.2.

###################

# NOTES TO SELF:

# Backpropagate the error.
# Also:  an extra term that penalizes how much information the latent representation contains, to encourage using concise codes for the datapoints and find underlying structure that explains the most data with the least bits.

# The 2 blue boxes in figure 4 (right side) for training time.  And then in testing time I sample from a (normally-distributed) random set of numbers to put into the decoder (figure 5, red box) to generate examples similar to the input.

###################



import math
import numpy as np # using matrix multiplication makes it much easier to handle different sizes of layers and weights
import time # to delay output for user readability



# FUNCTIONS, PART ONE:  (helper functions)

def sigmoid(x): # this keeps x within -1 and 1
    return (1 / (1 + math.exp(-x)) -0.5)*2 # -0.5)*2 so it can reach from -1 to +1 (not just 0 to 1)


def sigmoidElementWise(matrix):
    result = [sigmoid(element) for element in matrix]
    return result


def functionElementWise(function,matrix):
    result = [function(element) for element in matrix]
    return result


def createZerosArray(length):
    matrix = np.zeros(length)
    return matrix


def matrixMultiply(x,y):
    result = np.multiply(x, y)
    return result


def sumAcrossRows(matrix):
    return np.sum(matrix, axis=1)


def subtractElementWise(a,matrix):
    result = [a-element for element in matrix]
    return result


def addElementWise(a,matrix):
    result = [a+element for element in matrix]
    return result


def multiplyElementWise(a,matrix):
    result = [a*element for element in matrix]
    return result


def getZ():
    #z = sigmoidElementWise(np.sum(np.multiply(i,w),axis=1))
    global i, w1, z
    z = matrixMultiply( i, w1 )
    z = sumAcrossRows( z )
    z = functionElementWise( sigmoid, z )


def getG():
    global z, w2, g
    g = matrixMultiply( z, w2 )
    g = sumAcrossRows( g )
    g = functionElementWise( sigmoid, g )


def propagate():
    getZ()
    getG()


def printouts():
    global i, w1, z, w2, g
    print
    print 'input or answer =\n',i,'\n'
    print 'input weights =\n',w1,'\n'
    print 'latent layer =\n',z,'\n'
    print 'output weights =\n',w2,'\n'
    print 'guess =\n',g,'\n'



# VARIABLES:  i --(w1)--> z --(w2)--> g (vs. a=i)

i = np.matrix('1 1 1')

w1 = np.matrix('-1 -0.5 0 ; -0.7 0.1 -1')

#getZ()

w2 = np.matrix('0 -0.1 ; -1 0.2 ; -1 0')

#getG()

propagate() # i --(w1)--> z --(w2)--> g (vs. a=i)

print
print 'input or answer =\n',i,'\n'
time.sleep(0.5)
print 'input weights =\n',w1,'\n'
time.sleep(0.5)
print 'latent layer =\n',z,'\n'
time.sleep(0.5)
print 'output weights =\n',w2,'\n'
time.sleep(0.5)
print 'guess =\n',g,'\n'
print '-> goal:  get the guess to approach ~ input answer'
print

a = i

s = 0.5 # sensitivity to error



# FUNCTIONS, PART TWO:  (this function depends on the previous functions and variables)

def train(numOfIters): # i --(w1)--> z --(w2)--> g (vs. a=i)
    
    # declare all as global here so can use values initialized above this function
    global i, w1, z, w2, g, a
    
    for iter in range(numOfIters): # train by going through iterations
                # error:
        e = subtractElementWise( a, g )
        
        # calculate change in weights based on output error and input:
        dw2 = multiplyElementWise( s, multiplyElementWise( e, z ) )
        dw1 = multiplyElementWise( s, matrixMultiply( d, i ) )
        
        # update weights:
        w2 = addElementWise( dw2, w2 )
        w1 = addElementWise( dw1, w1 )
        
        # calculate "guess" from input and weights:
        propagate()
        
        # print out to let us see what's happening:    print 'e =', round(e,3)
        printouts()
    
print '-> goal:  get the guess to approach ~ input answer'
print 'answer = ', a



# "MAIN FUNCTION" that calls on the above functions and variables:

print'________________________________________________'

#train(5)

print'________________________________________________'