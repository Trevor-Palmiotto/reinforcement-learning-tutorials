# Lesson 0 — The Gymnasium API

**Goal:** understand the agent–environment loop as Gymnasium expresses it, and
build the `rollout` helper that every later lesson depends on.

There is no learning in this lesson. That is on purpose: before you can write
an algorithm that improves a policy, you need a policy, an environment, and a
way to measure. This lesson builds those three things.

## The formalism

RL problems are modelled as a **Markov Decision Process** (MDP), a 5-tuple
`(S, A, P, R, γ)`:

| Symbol | Name | In Gymnasium |
|---|---|---|
| `S` | state space | `env.observation_space` |
| `A` | action space | `env.action_space` |
| `P(s'\|s,a)` | transition dynamics | hidden inside `env.step` |
| `R(s,a,s')` | reward function | the `reward` returned by `env.step` |
| `γ` | discount factor | *yours*, not the environment's |

Two things are worth pausing on.

**The dynamics are hidden.** `P` exists but you cannot query it. This is what
makes the problem *reinforcement* learning rather than planning — you learn
from sampled experience, not from a model you were handed. (Lesson 5 is where
you sit on the other side of this wall and write `P` yourself.)

**γ belongs to you, not the environment.** The discount factor is part of your
*objective*, not part of the problem specification. The return is

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ... = Σ_{k=0}^∞ γ^k · r_{t+k+1}
```

`γ ∈ [0, 1)` makes that infinite sum converge and encodes how much you care
about the future. `γ = 0` is myopic; `γ → 1` is far-sighted but high-variance.

**The Markov property** is the assumption that `s_t` contains everything
relevant about the history — that `P(s_{t+1} | s_t, a_t)` is the same whether
or not you also knew `s_{t-1}, s_{t-2}, …`. CartPole satisfies this because the
observation includes velocities. A single Atari *frame* does not, which is why
DQN stacks four of them.

## The loop

```python
obs, info = env.reset(seed=0)
while True:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### `terminated` vs `truncated` — the distinction that causes the most bugs

| | Meaning | Value of the final state | Bootstrap? |
|---|---|---|---|
| `terminated` | The MDP reached a terminal state: pole fell, lander crashed, goal reached. | `V(s') = 0` **by definition** | **No** |
| `truncated` | An outside condition stopped the episode early — almost always a time limit. | `V(s')` is whatever it is | **Yes** |

Every TD update in this curriculum has the form

```
target = r + γ · (1 - terminated) · V(s')
```

Note it is `terminated`, **not** `terminated or truncated`. If you mask on
truncation too, you teach the agent that surviving to the time limit is worth
zero — actively punishing the exact behaviour you want on CartPole. Gymnasium
split these into two flags specifically because the old single `done` flag made
this bug nearly invisible.

## Spaces

```python
env.observation_space   # Box(-inf, inf, (4,), float32) for CartPole
env.action_space        # Discrete(2)
env.action_space.sample()
```

`Discrete(n)` means actions are integers `0..n-1`. `Box` is a continuous vector
with per-dimension bounds. Lessons 1–5 all use `Discrete` action spaces; a
`Box` *action* space would require a different family of algorithms (DDPG, SAC,
PPO with a Gaussian head) which this curriculum does not cover.

## Wrappers

A wrapper is an environment that wraps another environment and modifies part of
the interface. They compose, and they are the idiomatic way to do preprocessing
without touching the env or the agent:

```python
env = gym.make("CartPole-v1")
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)
env = gym.wrappers.RecordEpisodeStatistics(env)
```

`RecordEpisodeStatistics` injects an `episode` key into `info` on the final
step, which is where the `mean_length` numbers in later lessons come from.

## Your exercises

Both live in [`exercises.py`](exercises.py). They raise `NotImplementedError`
until you fill them in.

1. **`episode_return(env, policy, gamma)`** — run one episode and return the
   *discounted* return. Verify that with `gamma=1.0` it matches the undiscounted
   sum, and that with `gamma=0.99` on CartPole it saturates near `1/(1-γ) = 100`
   no matter how long the pole stays up.

2. **`ObservationScaler(env, low, high)`** — a wrapper mapping each observation
   dimension linearly from `[low, high]` into `[-1, 1]`. You must update
   `observation_space`, not just the values: downstream code reads the space to
   size its networks. This is the preprocessing DQN needs when observation
   dimensions have wildly different magnitudes.

Check yourself:

```bash
uv run pytest tests/test_lesson0.py
```

Compare against [`solution/exercises.py`](solution/exercises.py) once you have
something working — not before.

## Run the demo

```bash
uv run python -m rl_tutorial.lesson0_api.demo
```

This compares a random policy against a hand-coded heuristic on CartPole. The
heuristic — "push in the direction the pole is already falling", reacting to
angular *velocity* rather than angle — scores around 200 versus random's ~23,
without a single gradient step.

That number is worth sitting with. A one-line controller with no parameters
gets most of the way to solving CartPole. When your DQN in Lesson 3 hits 150
and you feel good about it, remember this line of code beat it. **Always
benchmark against both random and the best trivial policy you can think of** —
a surprising amount of published RL has been embarrassed by exactly this.

## Key takeaways

- `terminated` and `truncated` are different and the difference matters in every
  update rule you will write.
- γ is part of your objective, not the environment.
- Evaluate over *many* episodes with a fixed seed offset; a single episode's
  return is nearly meaningless.
- Always benchmark against a random policy.
