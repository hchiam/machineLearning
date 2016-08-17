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