# Reinforcement Learning (RL)

- action-value function = estimated reward for an action(-state) (may be recursive, depending on how you choose to calculate it).
- policy = "how to decide next action" = "at state x do action y".

- Monte Carlo method(s) in the context of RL = used in slightly more realistic situations when you don't know the environment = make random decisions re-evaluate estimates of values for rewards for state-action pairs.

  - Monte Carlo (MC) methods require knowing a sequence from start to end, which isn't always realistically computable or even possible.

- Temporal Differential (TD) learning = even more realistic than MC methods because you update at each step instead of awaiting an entire sequence:

  - 2 methods for TD: SARSA vs Q-learning:

    - SARSA doesn't explore (penalizes exploration) but sticks to a "safe"/greedy/"conservative" policy = "on-policy" method because it updates the best policy and only tries the best policy. It can avoid large risks, but can also miss out on the optimal path if that happens to require a large negative reward partway through.
    - Q-learning (AKA Sarsamax) is slower but explores (doesn't penalize exploration) and can adapt to fast-changing environments = "off-policy" method because it updates the best policy but still updates action-value pairs based on a randomly-explored policy = better to default Q-learning to than SARSA in my opinion because it can converge on the optimal policy, unless you specifically need to avoid the risk of a large negative reward during exploration.

  - TD learning is a value-based method (not a policy-based method).

  - but TD learning requires a discrete action space.

- Policy Gradient learning = even more realistic than TD methods because the action space is no longer discrete (but requires more computing power) and can even behave probabilistically. It's also more intuitive than TD methods because you learn the optimal policy directly (policy-based method) without needing to track a separate action-value function estimate (value-based method).
  - estimate the gradient of return by sampling episodes

- google [CIRL reinforcement learning in ML](https://www.google.com/search?q=cirl+reinforcement+learning+in+ml)
