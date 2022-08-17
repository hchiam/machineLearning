# 25 Misnomers and Confusing Technical Terms in ML

https://product.hubspot.com/blog/misnomers-and-confusing-terms-in-machine-learning

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

## models

1. ML model / algorithm
2. model drift = "concept drift" or "example drift" or "distribution drift"
3. black-box model = "black-box mapping"
4. non-parametric model = "non-parametric algorithm"

## optimization

1. learning rate = "step size" (for ϵ or maybe for ϵ▽E(w) entirely)
2. Stochastic Gradient Descent (SGD) = "Stochastic Approximation of General Gradient of at-this-point-Expected Descent (SAGGED)"
   - btw: stochastic vs random vs unpredictable vs chaotic: https://www.quora.com/What-is-the-difference-between-chaotic-systems-and-stochastic-systems
3. momentum = "friction coefficient"
4. backpropagation = "loss gradient backpropagation"

## losses and activations

1. cross-entropy = "error function"
2. softmax/hardmax/softplus = "softargmax" and "softrectifier" or "Boltzmann function"
3. softmax loss = "cross-entropy loss" or "_**softmax**_ activation on a dense layer followed by cross-entropy _**loss**_"

## neural networks

1. neural networks = "differentiable, parameterized geometric functions" or "functionchain" or "multi-layer modeling" or "chain train"
2. multi-layer perceptron (MLP) = "feed-forward neural network" or "stacked logistic regression" or "plain chain train"
3. input layer = "inputs"
4. hidden layer = "latent layers"

## deep learning

1. tensor = "multi-dimensional array" (AKA "holor")
2. convolutional layer (i.e. cross-correlation in library implementations)
3. deconvolutional layer = "transposed convolution"
4. Bidirectional Encoder Representations from Transformers (BERT) = "Non-directional Encoder Representations from Transformers (NERT)"
5. RNN gates:
   - forget gate = "remember gate" or "keep gate"
   - input gate = "write gate"
   - output gate = "read gate"
