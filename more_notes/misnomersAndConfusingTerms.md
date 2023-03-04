# 25 Misnomers and Confusing Technical Terms in ML (Machine Learning)

https://product.hubspot.com/blog/misnomers-and-confusing-terms-in-machine-learning

(See further sources at that ^ link above.)

Common vocab = convention != beginner-friendly. Sometimes misleading names.

## stats

1. multinomial distribution = actually is categorical distribution (ML/NLP likes to call it "multinomial distribution", e.g. in Python code)
   - bernoulli distribution = 2 categories, 1 trial = 1 coin flip
   - binomial distribution = 2 categories, n trials = n coin flips
   - categorical distribution = >2 categories, 1 trial = 1 dice roll
   - multinomial distribution = >2 categories, n trials = n dice rolls
2. inference = "prediction" (because in stats, an inference is more general, but ML has 2 types of "inference": training/learning time, and prediction time. Only during prediction time does ML call it "inference")
3. R^2 = "sqrt(R^4)" or just "d", but you might want to use a better statistic anyways. BTW R^2 can surprisingly(?) be negative if the model is worse than trivial baseline (`R^2` = `1 - sum[(y-f)^2] / sum[(y-y)^2]`).
4. multi-armed bandit = "many one-armed bandits" or "stateless reinforcement learning" or "adaptive tests". Like a bunch of slot machines with 1 arm each and different unknown probabilities of success, and you have to decide which, how many times, and in what order to play each slot machine. A subset of stochastic scheduling.
   - btw: stochastic vs random vs unpredictable vs chaotic: https://www.quora.com/What-is-the-difference-between-chaotic-systems-and-stochastic-systems
5. regression / logistic regression = "logistic model"
   - classification for categorical variable, regression for continuous variable
   - but logistic regression is used for classification because it actually outputs a continuous probability number output (stats analysis: continuous value, but ML goal: final classification output)
   - (btw: calibrated to 0-1 range for probability)

## so-called "models"

1. ML model / algorithm = both are ambiguous/fuzzy and overlap with each other.
2. model drift = "concept drift" or "example drift" or "distribution drift" (when the features/labels are drifting away from the training distributions, but the model fails to also drift away from the training distributions).
3. black-box model = "black-box mapping" (when we can view the logic from input to output, but the real issue is we can't easily map the logic to causal thinking).
4. non-parametric model = "non-parametric algorithm"(?)
   - "non-parametric" = unfixed, growable number of parameters (not "no" parameters).
   - "model" = data generation or estimator (in stats), or just estimator (in ML).

## optimization

1. learning rate = "step size" (for ϵ or maybe for ϵ▽E(w) entirely)
2. Stochastic Gradient Descent (SGD) = "Stochastic Approximation of General Gradient of at-this-point-Expected Descent (SAGGED)"
   - SGD = gradient descent, but based on a random minibatch sample of the training data, but because of that it's not always in the right direction at the current point, but still guarantees convergence.
   - btw: stochastic vs random vs unpredictable vs chaotic: https://www.quora.com/What-is-the-difference-between-chaotic-systems-and-stochastic-systems
3. momentum = "friction coefficient" to encourage quicker descent down a shallowly-inclined ravine between steep walls, and avoid simply oscillating between the steep walls. (originally named "momentum hyperparameter")
4. backpropagation = "loss gradient backpropagation"

## losses and activations

1. cross-entropy = ???
   - cross-entropy = like comparing similarity of 2 languages by using the minimal Huffman Coding based on letter frequencies of one language to encode the other language, i.e. one way to quantify similarity of letter frequencies: https://www.reddit.com/r/explainlikeimfive/comments/gwzwot/eli5_what_is_crossentropy_in_mlnlp/
2. softmax/hardmax/softplus = "softargmax" (for softmax) and "softrectifier" (for softplus) or "Boltzmann function"
3. softmax loss = a contraction of "_**softmax**_ activation on a dense layer followed by cross-entropy _**loss**_" = output after those 2 steps (https://towardsdatascience.com/additive-margin-softmax-loss-am-softmax-912e11ce1c6b). Or just call it "cross-entropy loss".

## neural networks

1. neural networks = "differentiable, parameterized geometric functions" or "functionchain" or "multi-layer modeling" or "chain train"
2. multi-layer perceptron (MLP) = "feed-forward neural network" or "stacked logistic regression" or "plain chain train". (Even "multi-layer multi-perceptron" isn't accurate, since MLPs are usually also non-linear/probabilistic, which a perceptron isn't.)
3. input layer = "inputs"
4. hidden layer = "latent layers"

## deep learning

1. tensor = "multi-dimensional array" (AKA "holor")
   - "tensor" has a more constrained definition in physics, but is generalized/looser in ML (just "multi-dimensional").
2. convolutional layer (i.e. library implementations actually use cross-correlation because it's simpler to implement, just flipped)
3. deconvolutional layer = actually "transposed convolution" (but ML made these 2 synonyms, which is different from the strict sense of deconvolution in math).
   - https://github.com/vdumoulin/conv_arithmetic#transposed-convolution-animations
4. Bidirectional Encoder Representations from Transformers (BERT) = "Non-directional Encoder Representations from Transformers (NERT)"
   - btw "transformers" use "attention", which enables weights that can change "focus" during runtime instead of being fixed at runtime, and enables focus on specific details depending on context (https://en.wikipedia.org/wiki/Attention_(machine_learning))
5. RNN gates: (LSTM networks are a type of RNN)
   - forget gate = "remember gate" or "keep gate"
   - input gate = "write gate"
   - output gate = "read gate"

<hr></hr>

## More words:

- how to remember precision vs recall/sensitivity: https://stats.stackexchange.com/questions/122225/what-is-the-best-way-to-remember-the-difference-between-sensitivity-specificity (TP = True Positive)
  - PREcision = % TP of PREdicted
  - REcall = % TP of REal
  - (Accuracy = TP And TN out of All)
