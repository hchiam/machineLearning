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

- learning rate = step size
  - ideal learning rate in 1D: 1/f(x)''
  - ideal learning rate in 2D: 1/Hessian() = 1/matrixOfSecondPartialDerivatives()
  - ideal learning rate for general convex functions = ???
