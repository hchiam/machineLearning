#2-layer neural net code from: http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
X = np.array([  [0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1] ])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):
    
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    
    # how much did we miss?
    l1_error = y - l1
    
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
    
    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1



#and now my test code:

# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity
i1 = 0
i2 = 1
w1 = -1
w2 = -1
a = 1
s = 0.1
for iter in range(5):
    g = i1*w1 + i2*w2
    e = abs(a-g)
    d = s*(a-g)/abs(a-g)
    w1 = w1 + d
    w2 = w2 + d
    print g, e, w1, w2, d
print g

# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity
i1 = 0
i2 = 1
w1 = -1
w2 = -1
a = 1
for iter in range(5):
    g = i1*w1 + i2*w2
    e = (a-g)
    dw1 = e*i1
    dw2 = e*i2
    w1 = w1 + dw1
    w2 = w2 + dw2
    print g, e, w1, w2, dw1, dw2
print g

