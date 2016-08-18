# modified version of neuralNet2.py --> now with a hidden layer! (and sensitivity)

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

s = 0.5

for iter in range(6): # train by going through iterations
    
    # calculate "guess" from input and weights:
    h1 = i1*w1h1 + i2*w2h1
    h2 = i1*w1h2 + i2*w2h2
    g = h1*wh1 + h2*wh2
    
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
    print 'i =', round(i1,3), round(i2,3)
    print 'w =', round(w1h1,3), round(w1h2,3), round(w2h1,3), round(w2h2,3), '(i --> h)'
    print 'h =', round(h1,3), round(h2,3)
    print 'w =', round(wh1,3), round(wh2,3), '(h --> g)'
    print 'g =', g, 'unrounded'
    print 'errors =', round(dwh1,3), round(dwh2,3), round(dw1h1,3), round(dw1h2,3), round(dw2h1,3), round(dw2h2,3)
    print

print 'GUESS = ', g # output final "guess"

# expected output "guess" should approximate the "answer" = 1