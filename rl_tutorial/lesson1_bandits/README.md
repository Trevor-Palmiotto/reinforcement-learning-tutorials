# Lesson 1 — Multi-Armed Bandits

**Goal:** isolate the exploration/exploitation tradeoff by removing everything
else, and write your first `gymnasium.Env`.

A bandit is an MDP with **one state**. There is no sequencing, no delayed
reward, no credit assignment — pull an arm, get a reward, repeat. Everything
that remains is the exploration problem, which is why bandits are where it is
studied.

## What a bandit is

The name is from slot machines — a "one-armed bandit." Picture a row of `k`
slot machines. Each has its own fixed, unknown average payout. You get `T`
pulls total, and you want to finish with as much money as possible.

```
      arm 0      arm 1      arm 2        ...     arm 9
     ┌─────┐    ┌─────┐    ┌─────┐              ┌─────┐
     │ ?   │    │ ?   │    │ ?   │              │ ?   │
     └──┬──┘    └──┬──┘    └──┬──┘              └──┬──┘
    pays ~0.2  pays ~1.5  pays ~-0.3          pays ~0.9
      (hidden)   (hidden)   (hidden)            (hidden)

   each pull: choose one arm, receive one noisy reward
```

You cannot see the averages. All you ever observe is the reward you got, and
even that is noisy — pull the best arm and you might still get a bad payout by
chance. The only way to learn anything is to pull arms and watch.

### Reading the notation

| Symbol | Spoken | Means |
|---|---|---|
| `k` | "kay" | how many arms (10 throughout this lesson) |
| `a` | "a" | an action — which arm you pulled, an integer `0..k-1` |
| `T` | "big tee" | the total number of pulls you get |
| `t` | "little tee" | which pull you are on right now, `1..T` |
| `q*(a)` | "q-star of a" | the **true** average payout of arm `a`. Fixed, and hidden from you. |
| `Q_t(a)` | "Q-t of a" | **your estimate** of `q*(a)`, as of step `t`. This is what your code stores. |
| `N(a)` | "n of a" | how many times you have pulled arm `a` so far |
| `r_t` | "r-t" | the reward you actually received on pull `t` |

The `*` in `q*` conventionally marks "the true/optimal thing" as opposed to
your estimate — the same convention reappears in Lesson 2 as `Q*`, the optimal
action-value function. **The whole game is making `Q` converge to `q*` while
spending as few bad pulls as possible getting there.**

## The core tension

To know which arm is best you must sample all of them (**exploration**); to
earn reward you should pull the one you currently believe is best
(**exploitation**). Every pull spent exploring is a pull not spent earning, but
every pull spent exploiting a wrong belief is worse.

### Regret — the scoreboard

**Regret** measures what your ignorance cost you: the reward you would have
collected had you known the best arm from the start, minus what you actually
collected.

```
Regret(T) = T · max_a q*(a)  −  Σ_{t=1}^T E[r_t]
            └─ what a genie  ─┘  └─ what you ──┘
               would collect        collect
```

Reading the pieces:

- `max_a q*(a)` is the true average of the **best** arm. (`max_a` means "the
  largest value over all choices of `a`".) Multiply by `T` for the genie's
  total.
- `Σ_{t=1}^T` is a sum over every pull from `t = 1` to `t = T` — the same sigma
  notation as Lesson 0's return.
- `E[r_t]` is the **expected** reward on pull `t` — "expected" meaning the
  average over the randomness, since any single pull is noisy.

Regret only ever goes up; the question is *how fast*. This is the real
difference between the algorithms:

| | regret at `t=1000` | at `t=4000` | growth when `t` doubles |
|---|---|---|---|
| ε-greedy (ε=0.1) | 226 | 706 | **1.83×** — nearly linear |
| UCB (c=2) | 149 | 222 | **1.21×** — sublinear |

(Measured with this code over 150 runs; you can reproduce it.)

**Linear regret** means you keep making mistakes at a constant rate forever —
double the time, double the loss. That is exactly what fixed-ε ε-greedy does,
because it never stops pulling random arms. **Logarithmic regret** means
mistakes become rarer as you learn, so the curve flattens. UCB achieves this.

## Estimating action values

`Q_t(a)` is our estimate of `q*(a)`. The obvious estimator is just the average
of the rewards we have seen from arm `a` so far.

The naive way is to keep a list of every reward and average it. That works but
wastes memory, and it is unnecessary — you can keep a *running* average that
updates in place. If `Q_n` is the average of the first `n−1` rewards and `R_n`
is the new one:

```
Q_{n+1} = Q_n + (1/n)·(R_n − Q_n)
```

Here `n` counts pulls **of this particular arm**, not total steps, so this is
`N(a)` from the table above. Worked through with rewards 1, 2, 6:

| pull | reward `R_n` | old `Q_n` | error `R_n − Q_n` | step `1/n` | new `Q_{n+1}` |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.0 | +1.0 | 1/1 | **1.0** |
| 2 | 2.0 | 1.0 | +1.0 | 1/2 | **1.5** |
| 3 | 6.0 | 1.5 | +4.5 | 1/3 | **3.0** |

Each new estimate is exactly the running mean (1, 1.5, 3.0) — no history
stored.

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
                  /  ln t  \
   Q(a)  +  c · \/ ------- 
                     N(a)

   └ what you  ┘   └ how unsure you are ┘
     believe
```

In plain code: `Q[a] + c * math.sqrt(math.log(t) / N[a])`.

The first term is exploitation — the arm's current estimated value. The second
is an **uncertainty bonus**, and reading it piece by piece:

- `N(a)` in the denominator: the more you have pulled an arm, the **smaller**
  its bonus. Well-understood arms stop getting a handicap.
- `ln t` in the numerator (the natural logarithm of the current step): this
  creeps upward as time passes, so an arm ignored for a long stretch slowly
  becomes attractive again. It grows *slowly* — `ln` doubles only when `t`
  squares — so this is a gentle nudge, not a reset.
- `c` scales how much you care about uncertainty at all. `c = 0` is pure
  greedy; larger `c` explores harder.

Concretely, with `c = 2` and `t = 100`:

| `N(a)` | bonus | meaning |
|---|---|---|
| 1 | 4.29 | tried once — huge benefit of the doubt |
| 10 | 1.36 | some evidence |
| 100 | 0.43 | well understood, bonus nearly gone |

This is **optimism in the face of uncertainty** made explicit: the agent
prefers arms that are either *good* or *poorly understood*, which is exactly
the right criterion. An arm only gets dropped once you are confident it is
bad — and "confident" is doing real work, since the bonus encodes how much
evidence you have. UCB achieves logarithmic regret.

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

## Common traps

The tests will name these if you hit them, but they are worth knowing up front
— each produces code that *looks* like it works.

**Seeding through the global `np.random`.** Your env must draw from
`self.np_random`, which `super().reset(seed=seed)` sets up. The global
generator ignores the seed you were handed, so two envs reset with the same
seed get different problems and nothing is reproducible.

**`np.argmax` for greedy selection.** It always returns the *first* maximum.
Since `Q` starts all-equal, the agent then tries arm 0 forever and never
explores — which silently destroys optimistic initialisation, whose entire
mechanism depends on moving between tied arms. Break ties uniformly at random.

**Overwriting instead of averaging.** `Q[a] = reward` rather than
`Q[a] += (reward - Q[a]) / N[a]`. The estimate then tracks only the most
recent pull, and on noisy rewards the agent chases whichever arm last got
lucky.

**Letting `N(a) == 0` divide.** See the UCB note above: `nan`/`inf` propagate
into `argmax` and the agent fixates on the lowest-numbered untried arm while
appearing to sweep correctly.

## Key takeaways

- `New ← Old + α·(Target − Old)` is the template for every update in this course.
- Constant `α` tracks a moving target; `1/n` computes an exact mean. RL targets
  move, so RL uses constant `α`.
- Uncertainty-directed exploration (UCB) beats undirected exploration (ε-greedy),
  because it spends its exploration budget where the ignorance actually is.
- Fixed ε has a performance ceiling. This is why Lesson 2 decays it.
