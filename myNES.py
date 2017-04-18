import numpy as np

def f(guess):
    error = np.subtract(answer,guess)
    return error

answer = np.random.randint(100,size=(1,3))
print('answer = ' + str(answer))

guess = np.random.randint(100,size=(1,3))
error = f(guess)

while not np.array_equal(error, np.matrix([0,0,0])):
    print(' guess = ' + str(guess) + '\t error = ' + str(error))
    guess += error
    error = f(guess)

print(' guess = ' + str(guess) + '\t error = ' + str(error))