# Reinforcement Learning (RL)

- state value function = estimated reward for a state.
- action value function = estimated reward for an action(-state) (may be recursive, depending on how you choose to calculate it).
- policy = "how to decide next action" = "at state x do action y".

- Monte Carlo method(s) in the context of RL = used in slightly more realistic situations when you don't know the environment = make random decisions re-evaluate estimates of values for rewards for state-action pairs.

  - Monte Carlo (MC) methods require knowing a sequence from start to end, which isn't always realistically computable or even possible.

- Temporal Differential (TD) learning = even more realistic than Monte Carlo methods because you update at each step instead of awaiting an entire sequence:
  - 2 methods for TD: SARSA vs Q-learning:
    - SARSA doesn't explore (penalizes exploration) but sticks to a "safe"/greedy/"conservative" policy = "on-policy" method because it updates the best policy and only tries the best policy. It can avoid large risks, but can also miss out on the optimal path if that happens to require a large negative reward partway through.
    - Q-learning (AKA Sarsamax) is slower but explores (doesn't penalize exploration) and can adapt to fast-changing environments = "off-policy" method because it updates the best policy but still updates action-value pairs based on a randomly-explored policy = better to default Q-learning to than SARSA in my opinion because it can converge on the optimal policy, unless you specifically need to avoid the risk of a large negative reward during exploration.
