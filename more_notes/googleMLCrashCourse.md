# ML Crash Course from Google

https://developers.google.com/machine-learning/crash-course

These are notes more for my own reminders and active learning, not a summary of the entire course. To do your own active learning, I encourage you to write your own summary notes of what's new to you.

- learning from _**EXAMPLES**_ (inputs/features) to _**predict**_ (outputs/inferences) like a _**scientist**_ (observations & stats)

- traditional programming: code; --> ML projects: representing/modeling features

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
- **mini-batch SGD** = batch size of 10-1000 chosen at random to calculate the next gradient (and the next step) = less noisy than SGD, and still more efficient than full-batch gradient descent. Why does it work? Because large data sets naturally are more likely to contain redundant information, which is helpful to smooth out noisy gradients, but at larger and larger scales start to stop helping as much.

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

</details>

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

  - try splitting known data into a training set and a test set (technically called the validation set, before the final "real" test set)

- naively: [training set size > test set size (typically 80:20 split)](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)

  - expect test set performance to be similar but slightly worse than training perf
  - make sure sample set is large enough to be statistically significant (rule of thumb I've found outside this course: n > 20, so >20 for training set and >20 for test set)
  - make sure training set is representative and test set is representative (e.g. 50% winter 50% summer represented in both training set and test set)
  - if test set performs surprisingly well or better than training set, check whether any test set data accidentally leaked into the training set
  - [it's dangerous to repeatedly tweak our hyperparameters based on results from the _same_ test set, because that causes us to implicitly fit to the particular test set](https://developers.google.com/machine-learning/crash-course/validation/check-your-intuition)

- but rather prefer this: [training set ("params" / 1 model) --> validation set (20%) ("hyperparams" / best model) --> test set ("final" model)](https://developers.google.com/machine-learning/crash-course/validation/video-lecture) - to avoid overfitting to the validation set

  - [practice training with validation sets + test sets](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/validation_and_test_sets.ipynb?hl=en)

    - setup: (california_housing_train.csv --> `train_df` = training set, california_housing_test.csv --> `test_df` = test set)

      <details>

      <summary>Click to expand code</summary>

      ```py
      # Import modules
      import numpy as np
      import pandas as pd
      import tensorflow as tf
      from matplotlib import pyplot as plt

      pd.options.display.max_rows = 10
      pd.options.display.float_format = "{:.1f}".format

      train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

      test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

      scale_factor = 1000.0

      # Scale the training set's label.
      train_df["median_house_value"] /= scale_factor

      # Scale the test set's label
      test_df["median_house_value"] /= scale_factor

      def build_model(my_learning_rate):
        """Create and compile a simple linear regression model."""
        # Most simple tf.keras models are sequential.
        model = tf.keras.models.Sequential()

        # Add one linear layer to the model to yield a simple linear regressor.
        model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

        # Compile the model topography into code that TensorFlow can efficiently
        # execute. Configure training to minimize the model's mean squared error.
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                      loss="mean_squared_error",
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])

        return model


      def train_model(model, df, feature, label, my_epochs,
                      my_batch_size=None, my_validation_split=0.1):
        """Feed a dataset into the model in order to train it."""

        history = model.fit(x=df[feature],
                            y=df[label],
                            batch_size=my_batch_size,
                            epochs=my_epochs,
                            validation_split=my_validation_split)

        # Gather the model's trained weight and bias.
        trained_weight = model.get_weights()[0]
        trained_bias = model.get_weights()[1]

        # The list of epochs is stored separately from the
        # rest of history.
        epochs = history.epoch

        # Isolate the root mean squared error for each epoch.
        hist = pd.DataFrame(history.history)
        rmse = hist["root_mean_squared_error"]

        return epochs, rmse, history.history

      print("Defined the build_model and train_model functions.")


      def plot_the_loss_curve(epochs, mae_training, mae_validation):
        """Plot a curve of loss vs. epoch."""

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")

        plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
        plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
        plt.legend()

        # We're not going to plot the first epoch, since the loss on the first epoch
        # is often substantially greater than the loss for other epochs.
        merged_mae_lists = mae_training[1:] + mae_validation[1:]
        highest_loss = max(merged_mae_lists)
        lowest_loss = min(merged_mae_lists)
        delta = highest_loss - lowest_loss
        print(delta)

        top_of_y_axis = highest_loss + (delta * 0.05)
        bottom_of_y_axis = lowest_loss - (delta * 0.05)

        plt.ylim([bottom_of_y_axis, top_of_y_axis])
        plt.show()

      print("Defined the plot_the_loss_curve function.")
      ```

      </details>

    - actually splitting out a validation set from the training data: (`validation_split = 0.2`)

      <details>

      <summary>Click to expand code</summary>

      ```py
      # The following variables are the hyperparameters.
      learning_rate = 0.08
      epochs = 30
      batch_size = 100

      # Split the original training set into a reduced training set and a
      # validation set.
      validation_split = 0.2

      # Identify the feature and the label.
      my_feature = "median_income"    # the median income on a specific city block.
      my_label = "median_house_value" # the median house value on a specific city block.
      # That is, you're going to create a model that predicts house value based
      # solely on the neighborhood's median income.

      # SHUFFLE THE TRAINING DATA BEFORE SPLITTING OUT A VALIDATION SET:
      # shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

      # Invoke the functions to build and train the model.
      my_model = build_model(learning_rate)
      # SHOULD USE shuffled_train_df INSTEAD OF train_df DIRECTLY:
      # epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature,
      #                                     my_label, epochs, batch_size,
      #                                     validation_split)
      epochs, rmse, history = train_model(my_model, train_df, my_feature,
                                          my_label, epochs, batch_size,
                                          validation_split)

      plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                          history["val_root_mean_squared_error"])
      ```

      </details>

      - if training set loss curve is different from validation set loss curve, then the data in the two sets are different (i.e. aren't similar) - e.g. in the case of this Colab, running `train_df.head(n=1000)` reveals that the data was sorted by `longitude`, so the split isn't totally random and could be biased if longitude affects the relationship of `total_rooms` to `median_house_value` - you gotta shuffle the data!

    - finally testing with the test set:

      ```py
      x_test = test_df[my_feature]
      y_test = test_df[my_label]
      results = my_model.evaluate(x_test, y_test, batch_size=batch_size)
      ```

      - the RMSE's should be similar:
        - `root_mean_squared_error` (for training set) printed at the end of the last time you ran `train_model`
        - `val_root_mean_squared_error` (for validation set) printed at the end of the last time you ran `train_model`
        - `root_mean_squared_error` (for test set) printed by `my_model.evaluate`

- tips for representing features well: (so that you can multiply by model weights)

  - number --> number
  - string/name --> one-hot encodings (can be represented compactly with [sparse representation](https://developers.google.com/machine-learning/glossary#sparse_representation) if needed: just a number for the position of where the one-hot encoding would be, but convert back to one-hot representation for training)
  - no magic values: don't use -1 to mean not available, instead add another param to indicate whether available, e.g. days_on_market_defined: 1.0
  - aim for clear values for easier reasoning and debugging (e.g. age in years, not in seconds since Unix epoch)
  - consider taking out extreme outliers that are verified to be incorrect, or clipping ("binning") outliers, e.g. all above 4.0 cap become treated as 4.0
  - consider the "binning trick" (range buckets) to map non-linearities without fancy tricks (use one-hot encoding), e.g. to group by range buckets of latitudes
  - know your data: visualize, debug data (dashboard/duplicates/missing), monitor

- why scale features: scaling features to use similar scales helps gradient descent converge faster, helps avoid the "NaN trap" (what I call "math imprecision propagation"), and helps avoid having the model focus too much on features simply because they happen have a significantly wider range in the raw data

- you don't have to scale to exactly the same range: -1/+1 is similar enough to -3/+3 but is significantly overshadowed by 5000/100000
