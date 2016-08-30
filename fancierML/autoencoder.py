import math
import time # to delay output for user readability



# FUNCTIONS, PART ONE:  (helper functions)

def sigmoid(x): # this keeps x within -1 and 1
    return (1 / (1 + math.exp(-x)) -0.5)*2 # -0.5)*2 so it can reach from -1 to +1 (not just 0 to 1)


def multiplyListsElementWise(lista,listb):
    result = [a*b for a,b in zip(lista,listb)]
    return result


def functionElementWise(function,matrix):
    result = [function(element) for element in matrix]
    return result


def getZ():
    global i, w1, z1, z2, z
    # get weighted contributions from input
    z1 = i[0]*w1[0][0] + i[1]*w1[0][1] + i[1]*w1[0][2]
    z2 = i[0]*w1[1][0] + i[1]*w1[1][1] + i[1]*w1[1][2]
    # keep z1 and z2 nodes of latent layer in range
    z1 = sigmoid(z1)
    z2 = sigmoid(z2)
    # put it back together into one layer
    z = [z1, z2]


def getG():
    global z1, z2, w2, g1, g2, g3, g
    # get weighted contributions from latent layer
    g1 = z1*w2[0][0] + z2*w2[0][1]
    g2 = z1*w2[1][0] + z2*w2[1][1]
    g3 = z1*w2[2][0] + z2*w2[2][1]
    # keep g1, g2, and g3 nodes of output "guess" layer in range
    g1 = sigmoid(g1)
    g2 = sigmoid(g2)
    g3 = sigmoid(g3)
    # put it back together into one layer
    g = [g1,g2,g3]


def printouts():
    global i, w1, z, w2, g
    print
    print 'input or answer =\n',i,'\n'
    print 'input weights =\n',w1,'\n'
    print 'latent layer =\n',z,'\n'
    print 'output weights =\n',w2,'\n'
    print 'guess =\n',g,'\n'



# VARIABLES:  i --(w1)--> z --(w2)--> g (vs. a=i)

i = [1,1,1]

w1 = [[-1, -0.5, 0],[ -0.7, 0.1, -1]]

getZ()

w2 = [[0, -0.1],[-1, 0.2], [-1, 0]]

getG()

# i --(w1)--> z --(w2)--> g (vs. a=i)

a = i

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
print 'answer =\n',a,'\n'
print

s = 0.5 # sensitivity to error



# FUNCTIONS, PART TWO:  (this function depends on the previous functions and variables)

def train(numOfIters): # i --(w1)--> z --(w2)--> g (vs. a=i)
    
    # declare all as global here so can use values initialized above this function
    global i, w1, z, w2, g, a
    
    for iter in range(numOfIters): # train by going through iterations
        
        print 'a=',a
        print 'g=',g
        
        e = []
        # error:
        for i in range(len(a)):
            e.append(a[i]-g[i])
    
        print 's=',s
        print 'e=',e
        print'z=',z
        
        # calculate change in weights based on output error and input:
        dw2 = multiplyElementWise( s, np.matrix([[e[0]*z[0], e[1]*z[0], e[2]*z[0]], [e[0]*z[1], e[1]*z[1], e[2]*z[1]]]))
        print 'dw2=', dw2
        dw1 = multiplyElementWise( s, matrixMultiply( d, i ) )
        
        # update weights:
        w2 = addElementWise( dw2, w2 )
        w1 = addElementWise( dw1, w1 )
        
        # calculate "guess" from input and weights:
        getZ()
        getG()
        
        # print out to let us see what's happening:    print 'e =', round(e,3)
        printouts()
    
print '-> goal:  get the guess to approach ~ input answer'
print 'answer =\n', a



# "MAIN FUNCTION" that calls on the above functions and variables:

print'________________________________________________'

#train(5)

print'________________________________________________'