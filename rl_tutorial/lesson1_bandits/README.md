# Lesson 1 — Multi-Armed Bandits

**Goal:** isolate the exploration/exploitation tradeoff by removing everything
else, and write your first `gymnasium.Env`.

A bandit is an MDP with **one state**. There is no sequencing, no delayed
reward, no credit assignment — pull an arm, get a reward, repeat. Everything
that remains is the exploration problem, which is why bandits are where it is
studied.

## The problem

`k` arms. Arm `a` pays a reward drawn from a fixed distribution with unknown
mean `q*(a)`. You pull `T` times. You want to maximise total reward.

The tension: to know which arm is best you must sample all of them
(**exploration**); to earn reward you should pull the one you currently believe
is best (**exploitation**). Every pull spent exploring is a pull not spent
earning, but every pull spent exploiting a wrong belief is worse.

The standard scoreboard is **regret** — what you lost by not knowing the answer
up front:

```
Regret(T) = T · max_a q*(a) − Σ_{t=1}^T E[r_t]
```

A good algorithm has regret growing like `log T`, not `T`. Linear regret means
you are still making a constant rate of mistakes forever, which is exactly what
fixed-ε ε-greedy does.

## Estimating action values

`Q_t(a)` is our estimate of `q*(a)`. The obvious estimator is the sample mean
of rewards seen from `a`. Computed incrementally, so you never store history:

```
Q_{n+1} = Q_n + (1/n)·(R_n − Q_n)
```

**Read this shape carefully — it is the whole course in one line.**

```
NewEstimate ← OldEstimate + StepSize · (Target − OldEstimate)
```

`(Target − OldEstimate)` is an **error**; the step size decides how far you
move to correct it. Q-learning (Lesson 2), DQN (Lesson 3) and actor-critic
(Lesson 4) are all this update with a different notion of "target". With
`1/n` you get the exact running mean. With a constant `α` you get an
exponentially-weighted average that tracks a *changing* `q*` — which is what
you want in real RL, where the target moves as the policy improves.

## The three strategies you will implement

### ε-greedy
With probability `ε` pull a uniformly random arm; otherwise pull `argmax_a Q(a)`.

Dead simple and hard to beat. Its flaw: exploration never stops and is never
targeted — at `ε = 0.1`, 10% of pulls go to a random arm forever, including
arms already known to be terrible. Linear regret.

### Optimistic initialisation
Set `Q_0(a) = 5` when true means are around 0, and act **greedily**.

No explicit exploration at all. Every arm looks wonderful, so the greedy agent
tries one, is disappointed (its estimate drops), and moves to another
still-untried arm — the optimism does the exploring. Elegant, and it beats
ε-greedy on this benchmark. But it is a *starting condition*, so it only drives
exploration early; it does nothing if the problem changes later.

### UCB (Upper Confidence Bound)
Pull the arm maximising

```
        ┌──────────┐
Q(a) + c·│  ln t    │
         │ ───────  │
         └  N(a)   ┘        i.e.  Q(a) + c · sqrt(ln t / N(a))
```

The second term is an **uncertainty bonus**: large for arms pulled rarely
(`N(a)` small), shrinking as evidence accumulates, and growing slowly with `t`
so that a long-neglected arm eventually gets revisited. This is *optimism in
the face of uncertainty* made explicit — the agent prefers arms that are either
good or poorly understood, which is precisely the right criterion. UCB achieves
logarithmic regret.

## Your exercises

In [`exercises.py`](exercises.py):

1. **`BanditEnv`** — a `gymnasium.Env` with `Discrete(k)` actions and a single
   dummy observation. This is your first environment implementation; Lesson 5
   does a much richer one.
2. **`EpsilonGreedyAgent`** — incremental sample-mean updates + ε-greedy action
   selection, with optional optimistic init and optional constant step size.
3. **`UCBAgent`** — the bonus above. Handle `N(a) = 0` by pulling every arm once
   first (the bonus is infinite there).

```bash
uv run pytest tests/test_lesson1.py
```

## Run the experiment

```bash
uv run python -m rl_tutorial.lesson1_bandits.experiment
```

This reproduces the Sutton & Barto §2 comparison: 2000 independent 10-armed
testbeds, 1000 steps each, averaged. Two panels — average reward, and **%
optimal action**, which is the more diagnostic of the two because it shows
directly whether the agent has *identified* the best arm rather than merely
stumbling into decent reward.

Things to look for in the output:

- Add `--steps 200` and watch the first 50 steps closely: the ordering there is
  the opposite of the final ordering.
- ε = 0.01 learns far more slowly (≈60% optimal at step 1000 versus ≈80% for
  ε = 0.1) but is still climbing steeply when ε = 0.1 has flattened. Extend the
  run with `--steps 6000` and you will see it reach ≈85%, essentially level with
  ε = 0.1 and still rising.

  This is the ceiling effect. A fixed ε caps you at `1 − ε + ε/k` optimal
  actions *forever* — 91% for ε = 0.1, 99.1% for ε = 0.01 — so the smaller ε
  must eventually win. "Eventually" is the catch: it takes many thousands of
  steps, and the default 1000-step plot shows only the part of the story where
  the larger ε looks better. **The horizon you plot determines which algorithm
  appears to win**, which is worth remembering every time you read an RL
  learning curve, including your own in later lessons.
- UCB rises fastest of all four over the first ~100 steps: it spends its first
  10 pulls trying each arm exactly once, then immediately concentrates on the
  plausible candidates rather than continuing to sample uniformly.
- Optimistic greedy starts *worst* — it dips below zero reward early, because
  every arm must be tried and found disappointing before the estimates become
  useful — and then overtakes everything. Early exploration is an investment
  that shows up as a loss on the reward panel before it pays.

## Key takeaways

- `New ← Old + α·(Target − Old)` is the template for every update in this course.
- Constant `α` tracks a moving target; `1/n` computes an exact mean. RL targets
  move, so RL uses constant `α`.
- Uncertainty-directed exploration (UCB) beats undirected exploration (ε-greedy),
  because it spends its exploration budget where the ignorance actually is.
- Fixed ε has a performance ceiling. This is why Lesson 2 decays it.
