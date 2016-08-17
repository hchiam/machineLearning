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