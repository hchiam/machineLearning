# My attempt to create a super simple (if over-simplified) version of a VAE (Variational Auto-Encoder).

# See http://arxiv.org/pdf/1606.05908v2.pdf for Carl Doersch's tutorial on VAE's.

# I personally recommend looking at Figure 4's right side (page 10) and Figure 5 (page 11) for the intuition behind a VAE:
# Blue blocks = to be optimized (minimize difference of output-input and minimize the "difference" of 2 probabilities).
# Red blocks = sampling from normalized distributions with mean & variance in brackets:  N(mean, variance).
# Also see explanation in Section 2.4.2.

# Backpropagate the error.
# Also:  an extra term that penalizes how much information the latent representation contains, to encourage using concise codes for the datapoints and find underlying structure that explains the most data with the least bits.

The 2 blue boxes in figure 4 (right side) for training time.  And then in testing time I sample from a (normally-distributed) random set of numbers to put into the decoder (figure 5, red box) to generate examples similar to the input.

