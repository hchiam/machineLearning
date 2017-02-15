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

# i=input, w=weight, g=guess, a=answer, e=error, d=direction to move g closer to a, s=sensitivity

i1 = 0 # input 1
i2 = 1 # input 2
w1 = -1 # weight 1
w2 = -1 # weight 2
a = 1 # the correct "answer"

s = 0.1 # sensitivity to error

for iter in range(21): # train by going through iterations
    g = i1*w1 + i2*w2 # calculate "guess" from input and weights
    
    e = abs(a-g) # error
    
    d = (a-g)/abs(a-g) # direction to move "guess" closer to "answer"
    
    w1 = w1 + d*s # update weight
    w2 = w2 + d*s # update weight
    print g, e, w1, w2, d*s # print out to let us see what's happening

print g # output final "guess"

# expected output "guess" should approximate the "answer" = 1