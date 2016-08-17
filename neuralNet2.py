# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity

i1 = 0 # input 1
i2 = 1 # input 2
w1 = -1 # weight 1
w2 = -1 # weight 2
a = 1 # the correct "answer"

for iter in range(3): # train by going through iterations
    g = i1*w1 + i2*w2 # calculate "guess" from input and weights
    
    e = (a-g) # error
    
    dw1 = e*i1 # calculate change in weight 1 based on output error and input 1
    dw2 = e*i2 # calculate change in weight 2 based on output error and input 2
    
    w1 = w1 + dw1 # update weight
    w2 = w2 + dw2 # update weight
    print g, e, w1, w2, dw1, dw2 # print out to let us see what's happening

print g # output final "guess"

# expected output "guess" should approximate the "answer" = 1