# ML Crash Course from Google

https://developers.google.com/machine-learning/crash-course

Quick link to exercises/quizzes: https://developers.google.com/machine-learning/crash-course/exercises

These are notes more for my own reminders and active learning, not a summary of the entire course. To do your own active learning, I encourage you to write your own summary notes of what's new to you.

- learning from _**EXAMPLES**_ (inputs/features) to _**predict**_ (outputs/inferences) like a _**scientist**_ (observations & stats)

- traditional programming: code; --> ML projects: representing/modeling features

- (rules + data) -> traditional programming -> (answers)
- (answers + data) -> machine learning -> (rules)

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
  - consider clipping ("binning") outliers, e.g. all above 4.0 cap become treated as 4.0 (otherwise take out extreme outliers only if they are verified to be incorrect)
  - consider the "binning trick" (range buckets) to map non-linearities without fancy tricks (use one-hot encoding), e.g. to group by range buckets of latitudes, because we expect a **_NON_**-linear relationship between latitude and housing prices.
    - binning enables learning different weights per latitude group
    - bins can be by whole numbers, or handle outliers with bins by quantile of the data
  - know your data: visualize, debug data (dashboard/duplicates/missing), monitor
    - think about what you'd expect the data to look like and see if it matches
    - if it doesn't match, see if you can explain why it doesn't match what you'd expect
    - double-check that the data agrees with other data sources
  - remove legitimately bad data: entry with missing data, duplicate entry, incorrect entry label, incorrect entry value (tip: generate most common values lists to see if they make sense, like does the most common language make sense)

- why scale features: scaling features to use similar scales helps gradient descent converge faster, helps avoid the "NaN trap" (what I call "math imprecision propagation"), and helps avoid having the model focus too much on features simply because they happen have a significantly wider range in the raw data

- you don't have to scale to exactly the same range: -1/+1 is similar enough to -3/+3 but is significantly overshadowed by 5000/100000

- **feature crosses** are a simple way to let you find or represent patterns that can't be done with linear y = b + w1 x1 + w2 x2, for example a XOR-like 2D map of x1 and x2 can be captured with y = b + w1 x1 + w2 x2 + w3 x3 where x3 = x1 x2. (Neural networks are like more sophisticated feature crosses.)

  - e.g.: x3 = x1 x2
  - e.g.: adding x1^2 and x2^2 terms can let you create a circle
  - e.g.: cost of 3 rooms at one city latitude (real-world locations have 2 dimensions! = feature cross of (_binned_) latitude and longitude, or consider postal code) >> 3 rooms at another city latitude
  - e.g.: tic-tac-toe sequence info >> tic-tac-toe position info
  - e.g.: country:usa AND language:spanish
  - e.g.: satisfaction <-- dog behaviour type X time of day
  - --> similar to **neural nets** = more complex/intense/black-box but more powerful (requires non-linear outputs from each neuron/node in order to produce non-linear output functions, the simplest being **ReLU**: **Re**ctified **L**inear **U**nit activation function = `max(0,x)`. Typically used as `max(0, wx + b)` (non-linear) <-- `y = mx + b` (linear))
  - colab: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/representation_with_a_feature_cross.ipynb?hl=en
  - check your understanding: https://developers.google.com/machine-learning/crash-course/feature-crosses/check-your-understanding
  - **caveat**: sparse feature crosses can be bad and unhelpful, e.g. search query words crossed with unique videos = model memory size explosion and extra noisy coefficients causing overfitting --> fix by making weights 0 with regularization (see notes below):
    - ❌ fix with **L0** regularization = penalize `count(non-zero weights)` --> but NP-hard problem to solve, hard to optimize (non-convex).
    - ✅ fix with **L1** regularization = relaxed version of L0: penalize `sum(abs(weights))` (and set weight = 0 when weight < 0) --> easier to optimize (convex), encourages sparsity, unlike L2. One downside of L1 regularization is it may also set weights to 0 for some informative features (features that are not non-informative but still weakly informative, or strongly informative on different scales, or strongly informative but strongly correlated to other similarly informative features)
    - ❌ fix with **L2** regularization = penalize `sum(squares(weights))` (details below). --> "squeezes" the weights towards zero, but won't actually "squash" them down to zero.
    - https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/check-your-understanding

- **dropout regularization** (see definition of regularization below) = randomly ignoring somewhere between 0% and 100% of the neurons in a neural network for a single gradient step (i.e. randomly pretend some of the neurons don't exist, and feed the data forward to the output once).

- **regularization** = penalizing model complexity (this is a better way to avoid overfitting i.e. improve model generalizability than setting some hard-to-do-in-practice "early stopping" point)

  - `complexity(model)` = one way is to prefer smaller weights:
    - in L2 regularization (AKA ridge): `complexity(model)` = `coefficient * sum(squares(weights))`.
      - so `loss` = `minimize( loss(data|model) + coefficient * sum(squares(weights)) )`
      - Recommended: large `coefficient` for smaller training data sets or when training and test sets look very different.

- **logistic regression**: generates probabilities output calibrated to 0-1 range (exclusive) for probability

  - can be implemented with sigmoid of the output to keep it between 0-1 = 0%-100%.
    - y' = 1 / (1+e^-z),
    - where z = b + w1 x1 + w2 x2 + ... wN xN.
  - loss can't use mean_squared_error but rather LogLoss.
    - LogLoss = sum[-y log(y') - (1-y)log(1-y')] across data points of (x,y) pairs,
      - where x are features,
      - where y is label output is either 0 or 1, and
      - where y' is the predicted value between 0-1.
  - **_to avoid overfitting with +/- infinity weights, logistic regression must use regularization_** (see notes above for regularization definition)
    - (see notes above for L2 regularization)
    - (see notes below for L1 regularization)
  - **_linear_ logistic regression** is very fast and _non-linear_ features can be extracted with feature crosses (see notes above)

- accuracy of 99% may not be good enough - you can get 99% by naively always saying one thing
- **accuracy** = correct predictions (+ve and -ve) / all predictions (+ve and -ve).
  - accuracy can be misleading because you sometimes actually care about _precision_ or _recall_ or both, especially _when there's an imbalance of positive/negative cases_:
  - _mnemonic:_ "a is for all, not just the positive ones" (vs precision).
  - Practice Accuracy vs Precision vs Recall: https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall
- **precision** = true positives / all predicted positives:
  - _intuition:_ precision = how much of the time was it reliably "crying wolf"?
  - _mnemonic:_ "p is for prophecy proven-ness".
  - _tends to:_ predict "wolf" only when absolutely sure beyond reasonable doubt.
  - _helpful over accuracy when:_ unbalanced classes
  - _unhelpful extreme:_ never cry "wolf", and none of your "cries" are disputable, because you made never "cried wolf".
  - _when it matters more than recall:_ precision matters more when false negatives are acceptable, e.g. spam filter allowing some spam through, but false positives are unacceptable, e.g. you wouldn't want any important emails going into the spam folder. **you don't want to be WRONG about detections**
- **recall** = true positives / all actual positives whether predicted or not:
  - _intuition:_ recall = how much of the actual "wolves" did it detect?
  - _mnemonic:_ "recall? wreck all? reckon all? recognize all?"
  - _tends to:_ predict a "wolf" any time hear the "bushes" move.
  - _helpful over accuracy when:_ unbalanced classes
  - _unhelpful extreme:_ always cry "wolf", and you'll catch all the "wolves".
  - _when it matters more than precision:_ recall matters more when false negatives are unacceptable, e.g. medical test results, and false positives are acceptable, e.g. positive COVID test is better safe than sorry. **you don't want to MISS out on any to detect**
- aside: https://towardsdatascience.com/should-i-look-at-precision-recall-or-specificity-sensitivity-3946158aace1
  - **recall** in ML (tp/all actual positives that exist = tp/(tp+fn)) = **sensitivity** in medicine (tp/all actual positives that exist = tp/(tp+fn))
  - **precision** in ML (tp/predicted positive = tp/(tp+fp)) = **_NOT THE SAME AS_** **specificity** in medicine (tn/all actual negatives that exist = tn/(tn+fp))
- **precision** and **recall** are often at odds with each other.
- **precision** and **recall** often need to be balanced.
- ask for both **precision** and **recall** of a model to evaluate a model.

- **ROC Curve** = graph of True Positive rates (y-axis) and False Positive rates (x-axis) at a bunch of decision threshold values (think spam probability threshold values)

  - **AUC** = area under that ^ ROC curve = probability that the model correctly assigns a higher probability to a random actually-positive example than it does to a random actually-negative example.

- **prediction bias** = when average prediction !== average observation --> something's wrong (incomplete feature set, biased training sample, buggy pipeline, etc.)

  - prediction bias is a good debug/sanity check
  - but if there's no prediction bias, there may still be problems with the model - test the model against other metrics

- **binary classification colab**: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/binary_classification.ipynb

  - **convert regression (continuous number) into classification (0/1) with threshold.**

- you can normalize datasets with feature ranges in very different ranges (e.g. 500-100000 vs 2-12) by turning numbers into Z-scores (i.e. number of standard deviations from mean = (original value - mean) / standard deviation).

- **python code for a simple neural net and a deep neural net**: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/intro_to_neural_nets.ipynb

  - import modules
  - load dataset
  - normalize values (features roughly same range)
  - represent features sensibly
  - plot
  - create model (add regularization as needed in the last code section of the colab)
  - train model

- visual explanation of **backpropagation** (just scroll): https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll

- backpropagation debugging:

  - backpropagation **exploding gradients**? consider **batch normalization** (or lowering the **learning rate**) by normalizing a layer's outputs before passing on to the next layer.
  - backpropagation **vanishing gradients**? consider **ReLU** activation functions to avoid constantly multiplying smaller and smaller numbers down to 0.
  - backpropagation **ReLU** outputs stuck at 0? consider lowering the **learning rate**.
  - or consider **dropout regularization** (see note earlier above)

- multi-class neural network = 1 output node per class:

  - one-vs-all: sigmoid output or softmax to require all outputs to sum up to 1 (100%)
    - [mnemonic: sigmoid is alphabetically before softmax, like how softmax is an extension of sigmoid: sigmoid takes one input and gives one output, softmax takes in a vector and outputs a vector](https://towardsdatascience.com/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9)
  - multi-class, single-label classification = can only be of one class: use one softmax loss for all possible classes
  - multi-class, multi-label classification = can be/have multiple classes: use one logistic regression loss for each possible class
  - note: when there's a lot of output classes to train for, consider an efficient strategy "Candidate Sampling" of calculating for all positive labels, but only a random sample of negatives
  - example colab for multi-class classification: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/multi-class_classification_with_MNIST.ipynb

- **embeddings** build on the previous notes. Each output neuron outputs a decimal number that can be used as a position along one "dimension", with the output layer of those neurons together producing a multi-dimensional vector representing an item in latent space (an embedding), which can then be compared to other embeddings to do interesting things like finding similarity. The output of those embedding representation neurons can then be fed into further neurons for further processing, like recommending similar items.

  - e.g. similarity of movies based on movies watched by users (train the bigger neural network by randomly using some of the movies watched as positive labels)
  - e.g. similarity of sentences with same meaning but using different words
  - e.g. some multi-dimensional representation of numbers that are the same but are written differently
    - (raw image bitmap -> sparse vector encoding like [1,3,999] instead of a huge bit array vector with lots of 0's -> n-dimensional embedding) + (other features) -> (extra hidden layers) -> (logit layer of exclusive probabilities for digits 0-9) -> (softmax loss: logit output layer versus one-hot target class label)
  - Embeddings are supposed to reduce dimensions (compared to your "vocab" size). A good rule of thumb for **number of dimensions for embeddings** = `4th_root(possible values like vocab size)`, but then validate and try it out for your use case.
  - embeddings can also be used to define and find similarity between diverse data types! (e.g. text, images, audio, etc.)
  - example embedding techniques: **PCA** to reduce dimensions, **word2vec** to map word similarity by their neighbouring words,

- **production ML systems**: more than just the ML model itself, things like data collection and data verification, deployment, static-offline/dynamic-online training, static-offline/dynamic-online inference, fairness/bias, etc.: https://developers.google.com/machine-learning/crash-course/production-ml-systems

  - _static-offline vs dynamic-online:_ simple and easier-to-verify but stale (but still monitor in case of seasonality/etc.) vs adapts but complex and must be able to monitor/rollback/quarantine/etc.
  - _data dependencies:_ minimal sources of data, input data availability, input data format versioning, ROI of extra input data, input data correlation, input data feedback loops (is the input data affected by the model's output, like stock market and social impact)
  - _bias:_
    - examples of cognitive bias affecting ML models: https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias
    - 3+1 flags you can check for in data: (missing, unexpected, skew (map it!)) https://developers.google.com/machine-learning/crash-course/fairness/identifying-bias + (confusion matrix of true/false positives/negatives, then confusion matrix true/false positives/negatives per demographic group) https://developers.google.com/machine-learning/crash-course/fairness/evaluating-for-bias
    - fairness colab: https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/intro_to_ml_fairness.ipynb
  - (and more things to consider) but thankfully many of the other pieces to production ML systems already have existing solutions you can re-use
  - check your understanding:
    - https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-training/check-your-understanding
    - https://developers.google.com/machine-learning/crash-course/static-vs-dynamic-inference/check-your-understanding
    - https://developers.google.com/machine-learning/crash-course/data-dependencies/check-your-understanding
    - https://developers.google.com/machine-learning/crash-course/fairness/check-your-understanding

- discover more colabs/notebooks at https://research.google.com/seedbank like [neural style transfer](https://aihub.cloud.google.com/p/products%2F7f7495dd-6f66-4f8a-8c30-15f211ad6957)
- discover more built-in 3rd party colab code snippets with the `< >` icon to open a search box to find things like 3rd-party visualizations

- read "rules of ML" to learn from others' experiences/mistakes: https://developers.google.com/machine-learning/guides/rules-of-ml
  - bookmarklet to pick a random ML tip to review: https://github.com/hchiam/learning-js/blob/main/bookmarklets/random-ml-tip-from-page.js

## more notes on terminology

https://github.com/hchiam/machineLearning/blob/master/more_notes/misnomersAndConfusingTerms.md

## practice real-life usage

https://github.com/hchiam/learning-tensorflow/tree/master/my_coursera_notes
