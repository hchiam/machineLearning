# ML Crash Course from Google

https://developers.google.com/machine-learning/crash-course

These are notes more for my own reminders and active learning, not a summary of the entire course. To do your own active learning, I encourage you to write your own summary notes of what's new to you.

- learning from _**EXAMPLES**_ (inputs/features) to _**predict**_ (outputs/inferences) like a _**scientist**_ (observations & stats)

- reminder-to-self:

  - regression = continuous value predictions (e.g. price, probability)
  - classification = discrete options predictions (e.g. yes/no, cat/dog/fish)

- **derivative**: df/dx = ratio of how much f changes when x changes
- **partial derivative**: 𝛅f/𝛅x = derivative with respect to only one variable x when there are other variables like (x, y, ...)
- **gradient of a function**: ▽f = ▽f(x,y) = (𝛅f/𝛅x, 𝛅f/𝛅y) = a vector that has partial derivatives for each variable or dimension = a vector where each dimension has a ratio of how much f changes when each of its variables changes (variable = dimension). ▽f = points in greatest increase. -▽f = points in greatest decrease = the direction you might want to go in to reduce error.

- learning rate = step size = multiplier for gradient, and is an example of a hyperparameter

  - ["Most machine learning programmers spend a fair amount of time tuning the learning rate."](https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate)
  - ideal learning rate in 1D: 1/f(x)''
  - ideal learning rate in 2D: 1/[Hessian()](https://en.wikipedia.org/wiki/Hessian_matrix) = 1/matrixOfSecondPartialDerivatives()
  - ideal learning rate for general convex functions = ???

- **SGD (Stochastic Gradient Descent)** = batch size of 1 chosen at random to calculate the next gradient (and the next step) = more efficient than full-batch gradient descent, but very noisy.
- **mini-batch SGD** = batch size of 10-1000 chose at random to calculate the next gradient (and the next step) = less noisy than SGD, and still more efficient than full-batch gradient descent. Why does it work? Because large data sets naturally are more likely to contain redundant information, which is helpful to smooth out noisy gradients, but at larger and large scales start to stop helping as much.

[NumPy refresher Colab UltraQuick Tutorial for ML](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb)

```py
import numpy as np
array_1d = np.array([1, 2, 3])
array_2d = np.array([[1, 2], [3, 4], [5, 6]])
array_sequence = np.arange(1, 5) # [1,2,3,4]
array_random_ints = np.random.randint(low=50, high=101, size=(6)) # 6 numbers in 50-100, exclusive of (101)
array_random_floats = np.random.random([6]) # array of 6 random floats
# "broadcasting" to auto-resize to arrays of compatible dimensions, like:
print(array_1d * 2 + 0.01) # [2.01, 4.01, 6.01]
```

[pandas refresher Colab UltraQuick Tutorial for ML](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)

[tf.keras linear regression with fake data Colab](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb)

[tf.keras linear regression with real data Colab](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb)
