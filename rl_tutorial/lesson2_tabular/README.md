# Lesson 2 — Tabular Q-Learning and SARSA

**Goal:** learn from *delayed* reward. Bandits gave you the reward immediately;
here an action's consequence may not show up for twenty steps.

This is where RL proper begins, and the two algorithms you build differ by a
single line — which makes the on-policy/off-policy distinction unusually
concrete.

## The value functions

The **action-value** function under policy π is the expected return from taking
`a` in `s` and following π thereafter:

```
Q^π(s,a) = E_π[ G_t | S_t = s, A_t = a ]
```

Its Bellman equation says a value equals the immediate reward plus the
discounted value of where you land:

```
Q^π(s,a) = E[ r + γ · Q^π(s', a') ]          a' ~ π(·|s')
```

The **optimal** action-value function `Q*` satisfies the Bellman *optimality*
equation, where the successor action is the best one rather than a sampled one:

```
Q*(s,a) = E[ r + γ · max_{a'} Q*(s', a') ]
```

Once you have `Q*`, the optimal policy is free: `π*(s) = argmax_a Q*(s,a)`.
This is why value-based RL learns values rather than policies — the policy
falls out of the values.

## Temporal-difference learning

We cannot compute those expectations (we do not have `P`). So we sample, and
apply the update template from Lesson 1:

```
Q(s,a) ← Q(s,a) + α · [ target − Q(s,a) ]
         └──────────────  δ, the TD error  ──┘
```

Everything hinges on what you use for `target`:

| Algorithm | Target | Policy it evaluates |
|---|---|---|
| **SARSA** | `r + γ·Q(s',a')` where `a'` is the action **actually taken next** | the behaviour policy (ε-greedy) — **on-policy** |
| **Q-learning** | `r + γ·max_{a'} Q(s',a')` | the greedy policy, regardless of behaviour — **off-policy** |

In code the difference is one line:

```python
# SARSA
target = reward + gamma * (1 - terminated) * q[next_state, next_action]
# Q-learning
target = reward + gamma * (1 - terminated) * q[next_state].max()
```

Note the `(1 - terminated)` mask — and note that it is `terminated`, **not**
`terminated or truncated`, exactly as Lesson 0 warned. A terminal state has
value 0 by definition; a time-limit truncation does not.

**"Bootstrapping"** means the target contains our own current estimate
`Q(s',·)`. We update a guess toward another guess. This sounds unstable and
sometimes is (Lesson 3 is largely about containing it), but it is what lets
learning happen every step rather than only at episode end.

## Why the difference matters: the cliff

The classic demonstration is Cliff Walking. A grid where the optimal path runs
along the edge of a cliff; falling in costs −100 and resets you.

- **Q-learning** learns `Q*`, whose greedy policy walks right along the cliff
  edge — the shortest path. But while *behaving* ε-greedily it keeps falling
  in, so its **online** reward is poor.
- **SARSA** evaluates the ε-greedy policy it is actually following. That policy
  sometimes takes a random action, and a random action next to a cliff is
  catastrophic — so SARSA's values account for the risk and it learns the
  **safer** path one row back, earning more online reward.

Neither is wrong. They answer different questions: Q-learning asks "what is
best if I act optimally?", SARSA asks "what is best given that I will keep
exploring?" If exploration has real costs — a physical robot — SARSA's question
may be the one you want.

Run it and see:

```bash
uv run python -m rl_tutorial.lesson2_tabular.cliff
```

## Convergence

Tabular Q-learning converges to `Q*` with probability 1 provided:

1. every state-action pair is visited infinitely often, and
2. the step sizes satisfy `Σα = ∞` and `Σα² < ∞`.

Condition 1 is why ε never decays fully to zero in our schedules. Condition 2
is formally violated by a constant `α` — but a constant `α` is used anyway in
practice because it tracks the non-stationary target better, and the resulting
small persistent noise is a price worth paying.

## Your exercises

In [`exercises.py`](exercises.py):

1. **`greedy_action` / `epsilon_greedy_action`** — with random tie-breaking, as
   in Lesson 1.
2. **`td_target`** — the one function that distinguishes the two algorithms.
   Take an `algo` argument of `"q_learning"` or `"sarsa"`.
3. **`train`** — the shared loop. Both algorithms use it; only `td_target`
   differs. Note the ordering subtlety: SARSA needs `a'` *before* it can
   update, so the loop must choose the next action before updating, then reuse
   it as the next iteration's action.

```bash
uv run pytest tests/test_lesson2.py
```

## Run the experiments

```bash
# FrozenLake, slippery and not
uv run python -m rl_tutorial.lesson2_tabular.train_frozenlake

# The cliff — where the two algorithms visibly disagree
uv run python -m rl_tutorial.lesson2_tabular.cliff
```

`FrozenLake-v1` with `is_slippery=True` moves you in your intended direction
only 1/3 of the time. This makes it a genuinely stochastic MDP and a good test
of whether your update handles expectation correctly — a deterministic-looking
implementation will still learn on the non-slippery version and fail here.

## Key takeaways

- The TD error `δ = r + γ·V(s') − V(s)` is the learning signal for everything
  that follows.
- On-policy vs off-policy is one line of code and a real behavioural difference.
- Mask bootstrapping with `terminated`, never with `truncated`.
- Tabular methods need a table of size `|S| × |A|`. FrozenLake has 64 states;
  CartPole has infinitely many. That wall is what Lesson 3 breaks through.
