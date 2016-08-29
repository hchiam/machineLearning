# My attempt to create a super simple (if over-simplified) version of a VAE (Variational Auto-Encoder).

# See http://arxiv.org/pdf/1606.05908v2.pdf for Carl Doersch's tutorial on VAE's.

# I personally recommend looking at Figure 4's right side (page 10) and Figure 5 (page 11) for the intuition behind a VAE:
# Blue blocks = to be optimized (minimize difference of output-input and minimize difference of 2 probabilities).
# Red blocks = sampling from normalized distributions with mean & variance in brackets:  N(mean, variance).

# Equation 5 (as of August 2016 version) is the key equation to understand the mathematical soundness of the procedure.

