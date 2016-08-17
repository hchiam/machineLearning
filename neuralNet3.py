# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity

i1 = 0 # input 1
i2 = 1 # input 2
w1 = -1 # weight 1
w2 = -1 # weight 2
a = 1 # the correct "answer"

s = 0.1 # sensitivity to error

for iter in range(52): # train by going through iterations
    g = i1*w1 + i2*w2 # calculate "guess" from input and weights
    
    e = abs(a-g) # error
    
    d = (a-g)/abs(a-g) # direction to move "guess" closer to "answer"
    
    dw1 = d*e*i1 # calculate change in weight 1 based on output error and input 1
    dw2 = d*e*i2 # calculate change in weight 2 based on output error and input 2
    
    w1 = w1 + dw1*s # update weight
    w2 = w2 + dw2*s # update weight
    
    if iter%10 ==0: # every 10 iterations:
        print g, e, w1, w2, dw1*s, dw2*s, d # print out to let us see what's happening

print g # output final "guess"