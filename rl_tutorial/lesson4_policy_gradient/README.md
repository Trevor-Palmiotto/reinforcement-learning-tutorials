# Lesson 4 — Policy Gradients

**Goal:** optimise the policy directly, instead of deriving it from values.

Everything so far learned `Q` and read the policy off it with `argmax`. That
has real limits: the `argmax` is awkward with continuous actions, the policy
changes discontinuously as values cross over, and a stochastic optimal policy
cannot be represented at all. Here we parameterise the policy itself,
`π_θ(a|s)`, and do gradient *ascent* on its performance.

## The policy gradient theorem

We want to maximise `J(θ) = E_{τ~π_θ}[G(τ)]`. The gradient looks impossible —
the distribution we are averaging over depends on `θ`. The trick is the
score-function identity, `∇p = p·∇log p`:

```
∇J(θ) = E_π [ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]
```

Read it as a weighted maximum-likelihood objective. `∇log π(a|s)` is the
direction that makes action `a` more likely in state `s`; `G_t` is a scalar
saying how much we liked what followed. **Take good actions and make them more
likely, in proportion to how good.** No dynamics model, no bootstrapping.

Since optimisers minimise, the loss we hand to torch is

```
L(θ) = − Σ_t log π_θ(a_t|s_t) · G_t
```

and this is why the implementation is only a few lines: build a `Categorical`
over the logits, call `.log_prob(a)`, multiply by the return, negate, mean.

## REINFORCE and its problem

REINFORCE is the above, using the full Monte-Carlo return `G_t` computed after
the episode ends. It is **unbiased** — and has enormous **variance**. `G_t`
depends on every random choice made over the whole remaining episode, so the
same state-action pair can be credited wildly different values from one episode
to the next. Training is slow and erratic.

Two standard fixes.

### 1. A baseline

Subtract any function of state `b(s)` from the return:

```
∇J = E[ Σ_t ∇log π(a_t|s_t) · (G_t − b(s_t)) ]
```

This leaves the gradient **unbiased** — `E[∇log π · b(s)] = 0`, because
`b(s)` is constant with respect to the action and the score function has zero
mean — while potentially slashing variance.

Why it helps intuitively: on CartPole every return is positive, so plain
REINFORCE increases the probability of *every* action ever taken, and only the
relative magnitudes eventually sort things out. With `b(s) = V(s)` the weight
becomes positive only for better-than-average actions, and negative for worse.
That is a far cleaner signal.

`V(s)` is learned by regression on the observed returns — a second network head
trained with MSE.

### 2. Bootstrapping the return (actor-critic)

`G_t − V(s_t)` still uses the full Monte-Carlo return. Bootstrap it instead,
looking ahead `n` steps and then trusting the critic:

```
A_t = r_t + γ·r_{t+1} + … + γ^(n−1)·r_{t+n−1} + γ^n·V(s_{t+n}) − V(s_t)
```

This is the **advantage**: how much better `a_t` was than the average action in
`s_t`. Small `n` leans on the critic (low variance, high bias); large `n` uses
more real reward (low bias, higher variance). `n → ∞` recovers `G_t − V(s_t)`.

| | Return estimate | Bias | Variance |
|---|---|---|---|
| REINFORCE | `G_t` | none | very high |
| + baseline | `G_t − V(s_t)` | none | high |
| actor-critic (`n`-step) | `Σγ^k r + γ^n V(s_{t+n}) − V(s_t)` | tunable | tunable |

**`n` is not a detail — on CartPole it dominates everything else.** Measured
with this code at 400 episodes, everything else held fixed:

| `n` | final mean-100 return |
|---|---|
| 1 | ~115 |
| 5 | ~262 |
| 20 | ~480 |

The one-step advantage (`n = 1`) is *worse than plain REINFORCE*. That looks
like a bug and is not one. CartPole pays a constant reward of 1 per step, so
`r + γV(s') − V(s)` is almost entirely determined by the critic's own error —
until `V` is already accurate, the advantage carries nearly no signal about
which action was better. Longer lookahead injects real reward and the signal
returns.

This is worth dwelling on, because "add a critic and bootstrap" is usually
presented as a straight upgrade. It is a **trade**, and picking the wrong point
on the dial is worse than not trading at all. GAE (what PPO uses) interpolates
smoothly across this dial instead of picking one `n`.

## On-policy, and what it costs

The expectation is over trajectories from `π_θ`. Once you update `θ`, previously
collected data is off-policy and formally invalid. So policy gradient methods
throw data away after one update — no replay buffer, and far worse sample
efficiency than DQN. (PPO recovers some of this with importance ratios and a
clipped objective.)

This is why Lesson 3's replay buffer has no counterpart here, and it is the
main practical trade-off between the two families.

## Implementation notes that matter

**Normalise returns.** Standardising `(G − mean)/std` across the batch is a
crude but very effective variance reduction. It technically biases the
gradient; everyone does it anyway.

**Detach the advantage in the policy loss.** The advantage is a *weight*, not
something to differentiate through. If gradients flow from the policy loss into
`V`, you are training the critic to make the actor's loss small rather than to
predict returns. Use `advantage.detach()`.

**Compute returns backwards.** `G_t = r_t + γ·G_{t+1}`, iterating from the end
of the episode, is `O(T)`. A forward double loop is `O(T²)`.

**Entropy bonus.** Subtracting `β·H(π)` from the loss discourages the policy
from collapsing to determinism too early. This is the policy-gradient analogue
of ε-greedy — exploration comes from the policy's own stochasticity.

## Your exercises

In [`exercises.py`](exercises.py):

1. **`compute_returns`** — discounted returns, backwards, with the terminal
   mask.
2. **`PolicyNetwork`** — outputs **logits**, not probabilities. Let
   `Categorical(logits=...)` handle the softmax: it is numerically stable in a
   way that `log(softmax(x))` is not.
3. **`reinforce_loss`** — the score-function loss, with optional baseline and
   entropy bonus.
4. **`nstep_advantages`** — the bias/variance dial itself. Watch the two edge
   cases: a window that hits a terminal step must not bootstrap, and one that
   runs off the end of the episode must not read past it.
5. **`actor_critic_loss`** — advantages plus the critic's regression loss,
   returning the parts separately so you can diagnose which half is failing.

```bash
uv run pytest tests/test_lesson4.py
```

## Run it

```bash
uv run python -m rl_tutorial.lesson4_policy_gradient.train              # all three
uv run python -m rl_tutorial.lesson4_policy_gradient.train --variance   # multi-seed
uv run python -m rl_tutorial.lesson4_policy_gradient.train --nstep      # the n dial
```

Run `--nstep` first. It is the most instructive of the three.

The `--variance` run trains each variant across several seeds and plots the
median with an inter-quartile band. This is the point of the lesson: the
variance-reduction claim is something you should *see*, not take on faith. The
bands should narrow visibly from REINFORCE to baseline to actor-critic.

It also reports the standard deviation of returns across seeds — the single
number that makes the argument.

## Key takeaways

- Policy gradient = weighted maximum likelihood: make good actions more likely.
- A state-dependent baseline reduces variance without adding bias.
- Bootstrapping trades bias for variance, and the exchange rate (`n`) matters
  more than almost any other hyperparameter here.
- On-policy methods cannot reuse data, which is their central cost.
- Always plot across seeds. In RL a single run can show almost anything.
