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

<details>

<summary>Click to expand code</summary>

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

</details>

[NumPy + pandas refresher Colab UltraQuick Tutorial for ML](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)

- DataFrame = array with named columns and numbered rows

<details>

<summary>Click to expand code</summary>

```py
import numpy as np
import pandas as pd
array_2d = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
# array_2d = np.random.randint(low=0,high=101,size=(5,2))
column_names = ['temperature', 'activity']
dataframe = pd.DataFrame(data=array_2d, columns=column_names)
#    temperature  activity
# 0            0         3
# 1           10         7
# 2           20         9
# 3           30        14
# 4           40        15
dataframe['new column name'] = dataframe['activity'] + 2
#    temperature  activity  new column name
# 0            0         3                5
# 1           10         7                9
# 2           20         9               11
# 3           30        14               16
# 4           40        15               17
rows_0_to_2 = dataframe.head(3)
rows_0_to_2 = dataframe[0:3]
row_2 = dataframe.iloc[[2]]
row_2 = dataframe[2:3]
rows_2_to_4 = dataframe[2:5]
temp_column = dataframe['temperature']
cell_row_0_temp = dataframe['temperature'][0]
independent_clone = pd.DataFrame.copy(dataframe) # not affected by changes in the original dataframe
```

</details>

[tf.keras linear regression with fake data Colab](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_synthetic_data.ipynb)

- try smaller learning rates (AKA "step sizes")
- try larger epochs (i.e. "train more")
- try smaller batches (i.e. "can't / don't need to check against all examples, per model update")

<details>

<summary>Click to expand code</summary>

```py
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
def build_model(step_size): # learning rate = step size
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=step_size),
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model
def train_model(model, feature, label, epochs, batch_size):
  training_history = model.fit(x=feature,
                               y=label,
                               batch_size=batch_size,
                               epochs=epochs)
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]
  epochs = training_history.epoch
  epoch_history = pd.DataFrame(training_history.history)
  error_history = epoch_history['root_mean_squared_error']
  return trained_weight, trained_bias, epochs, error_history
def plot_model(trained_weight, trained_bias, feature, label):
  plt.xlabel('feature')
  plt.ylabel('label')
  plt.scatter(feature, label)
  def model_red_line():
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_weight * x1 + trained_bias
    plt.plot([x0, x1], [y0, y1], c='r')
  model_red_line()
  plt.show()
def plot_loss_history(epochs, error_history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.plot(epochs, error_history, label='Loss')
  plt.legend()
  plt.ylim([error_history.min()*0.97, error_history.max()])
  plt.show()
def learn_and_plot(features, labels, step_size, epochs, batch_size):
  model = build_model(learning_rate)
  trained_weight, trained_bias, epochs, error_history = train_model(
    model,
    features,
    labels,
    epochs,
    batch_size)
  plot_model(trained_weight, trained_bias, features, labels)
  plot_loss_history(epochs, error_history)

features = ([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
labels   = ([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

learning_rate = 0.2 # AKA step size
epochs = 100
batch_size = 2 # minibatches are faster than using all 12 examples per model update

learn_and_plot(features, labels, learning_rate, epochs, batch_size)
```

<details>

[tf.keras linear regression with real data Colab](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_with_a_real_dataset.ipynb)

- look at stats --> be wary of features with anomalies (e.g. max is way above what you'd expect given the quartiles for 25%, 50%, 75%, to estimate 100% = max)
- look at stats --> correlation matrix might help find best-correlated features
- consider combining features, like population and rooms --> population density

<details>

<summary>Click to expand code</summary>

```py
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
def adjust_report_granularity():
  pd.options.display.max_rows = 10
  pd.options.display.float_format = '{:.1f}'.format
def get_csv_into_dataframe():
  dataframe = pd.read_csv(filepath_or_buffer='./california_housing_train.csv')
  # dataframe = pd.read_csv(filepath_or_buffer='https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv')
  dataframe['median_house_value'] /= 1000.0 # scale the data to keep features in similar number range
  return dataframe
adjust_report_granularity()
dataframe = get_csv_into_dataframe()
first_few_rows = dataframe.head()
column_stats = dataframe.describe()

def build_model(step_size): # learning rate = step size
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=step_size),
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return model
def train_model(model, feature, label, epochs, batch_size):
  training_history = model.fit(x=feature,
                               y=label,
                               batch_size=batch_size,
                               epochs=epochs)
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]
  epochs = training_history.epoch
  epoch_history = pd.DataFrame(training_history.history)
  error_history = epoch_history['root_mean_squared_error']
  return trained_weight, trained_bias, epochs, error_history
def plot_model(trained_weight, trained_bias, feature, label, dataframe):
  plt.xlabel(feature)
  plt.ylabel(label)
  random_examples = dataframe.sample(n=200)
  plt.scatter(random_examples[feature], random_examples[label])
  def model_red_line():
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_weight * x1 + trained_bias
    plt.plot([x0, x1], [y0, y1], c='r')
  model_red_line()
  plt.show()
def plot_loss_history(epochs, error_history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.plot(epochs, error_history, label='Loss')
  plt.legend()
  plt.ylim([error_history.min()*0.97, error_history.max()])
  plt.show()
def predict(n, feature, label, model, dataframe):
  batch = dataframe[feature][10000:10000 + n]
  predicted_values = model.predict_on_batch(x=batch)
  print('feature   label          predicted')
  print('  value   value          value')
  print('          in thousand$   in thousand$')
  print('--------------------------------------')
  for i in range(n):
      print('%5.0f %6.0f %15.0f' % (dataframe[feature][10000 + i],
                                    dataframe[label][10000 + i],
                                    predicted_values[i][0]))
def learn_plot_predict(feature_name, label_name, step_size, epochs, batch_size, dataframe):
  model = build_model(step_size)
  trained_weight, trained_bias, epochs, error_history = train_model(
    model,
    dataframe[feature_name],
    dataframe[label_name],
    epochs,
    batch_size)
  plot_model(trained_weight, trained_bias, feature_name, label_name, dataframe)
  plot_loss_history(epochs, error_history)
  predict(10, feature_name, label_name, model, dataframe)

learning_rate = 0.2 # AKA step size
epochs = 10
batch_size = 10 # minibatches are faster than using all 12 examples per model update

dataframe["pop_density"] = dataframe['population'] / dataframe['total_rooms']

# feature_name = 'pop_density'
feature_name = 'median_income' # has 0.7 correlation with median_house_value
label_name = 'median_house_value' # we'll predict house value based on pop density

learn_plot_predict(feature_name, label_name, learning_rate, epochs, batch_size, dataframe)

# still not great - can we use stats to help us see which feature correlate with the label? use a correlation matrix!:
correlation_matrix = dataframe.corr() # 1.0 = perfect, 0 = none, -1.0 = reverse perfect
```

</details>

- ML 3 basic assumptions which affect generalizing: (may be violated in practice)
  1. we get examples independently and identically ("i.i.d")
  2. the distribution is stationary
  3. we get examples from the same distribution
- remember the model is only getting samples of the true distribution (think like a scientist) - keep model as simple as possible so doesn't overfit to peculiarities of just what it was trained on --> easier/likelier to generalize to unseen examples
  - try splitting known data into a training set and a test set (before the "real" test set)
