# Reinforcement Learning, by Building It

A six-lesson curriculum that teaches reinforcement learning and Gymnasium at
the same time, by making you implement the concepts rather than read about
them.

Each lesson is a Python package containing:

- a **README** with the governing equations and how they map to the code,
- an **`exercises.py`** of scaffolded stubs with detailed docstrings, which
  raise `NotImplementedError` until you fill them in,
- a **`solution/`** reference implementation to compare against *afterwards*,
- a **runnable experiment** that produces numbers and a plot,
- **tests** that check your version and the reference against the same suite.

## Setup

All commands are run from this directory (`Reinforcement-Learning/`), which is
a standalone uv project with its own `.venv`.

```bash
uv sync
```

Verify:

```bash
uv run python -m rl_tutorial.lesson0_api.demo
```

## The lessons

Work through them in order; each builds on the last.

| # | Lesson                                                       | Concepts                                                               | Environment              |
| - | ------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------ |
| 0 | [The Gymnasium API](rl_tutorial/lesson0_api/)                 | MDPs, the agent loop, spaces,`terminated` vs `truncated`, wrappers | CartPole                 |
| 1 | [Multi-Armed Bandits](rl_tutorial/lesson1_bandits/)           | Exploration vs exploitation, incremental value estimation, UCB         | your own`BanditEnv`    |
| 2 | [Q-Learning and SARSA](rl_tutorial/lesson2_tabular/)          | TD learning, bootstrapping, on-policy vs off-policy                    | FrozenLake, CliffWalking |
| 3 | [Deep Q-Networks](rl_tutorial/lesson3_dqn/)                   | Function approximation, replay, target networks, Double DQN            | CartPole                 |
| 4 | [Policy Gradients](rl_tutorial/lesson4_policy_gradient/)      | REINFORCE, baselines, n-step actor-critic, bias/variance               | CartPole                 |
| 5 | [Build Your Own Environment](rl_tutorial/lesson5_custom_env/) | The`Env` contract, registration, reward shaping                      | your own GridWorld       |

Lesson 5 closes the loop: you train Lesson 2's agent, unmodified, on the
environment you wrote.

## How to work through a lesson

1. Read the lesson's `README.md`.
2. Open `exercises.py` and fill in the stubs. The docstrings tell you what the
   function must do and flag the specific mistakes that are easy to make.
3. Run the tests: `uv run pytest tests/test_lessonN.py`.
   Tests against unimplemented stubs *skip*, so a clean early run means "not
   done yet", not "passing".
4. Run the experiment and look at the plot.
5. *Then* read `solution/` and compare.

Every runnable script takes `--mine` to use your implementation instead of the
reference, so you can run the full experiment with your own code.

## Running the tests

```bash
# Everything except the training runs (a few seconds)
uv run pytest tests -m "not slow"

# Including end-to-end training (several minutes)
uv run pytest tests
```

## Shared utilities

[`rl_tutorial/common/`](rl_tutorial/common/) holds the pieces every lesson
reuses, so they are written once:

- `rollout.py` — `rollout()` and `evaluate()`, plus the `Transition` type.
- `schedules.py` — constant, linear and exponential ε decay.
- `plotting.py` — learning curves and multi-seed bands, on a colourblind-safe
  palette with a hard cap on series count.

## A few things this curriculum insists on

These come up in every lesson because they are what separates working RL code
from code that looks right and silently is not.

**`terminated` is not `truncated`.** A terminal state has value zero by
definition; a time-limit truncation does not. Every bootstrap in this
curriculum masks on `terminated` alone. Conflating them is the most common
bug in hand-written RL, and it usually presents as mediocre performance rather
than an error.

**Always benchmark against something trivial.** Lesson 0's one-line CartPole
heuristic scores ~200 with no learning at all. Several published results have
been embarrassed by exactly this comparison.

**One seed proves nothing.** Between-seed variance in RL routinely exceeds the
difference between algorithms. Lesson 4's `--variance` mode and
`plot_seed_band` exist for this reason. While building Lesson 3, an identical
DQN configuration scored 9.3 on one seed and 401.9 on another — which is also
why the training loop keeps the best checkpoint rather than the final weights.

**Run `check_env` on anything you write.** It catches in seconds the class of
environment bug that otherwise looks like an agent bug for an afternoon.

## Where the code disagrees with the textbook

Three of the experiments here produced results that contradict the usual
presentation, and the lessons were rewritten around the measurements rather
than the other way round. They are flagged in the relevant READMEs:

- **One-step actor-critic is *worse* than REINFORCE on CartPole** (Lesson 4).
  Not a bug — a constant per-step reward leaves a one-step advantage with
  almost no signal. The fix is n-step returns, and the sweep across `n` is the
  most instructive experiment in the curriculum.
- **Naive reward shaping does not break this GridWorld** (Lesson 5), even at
  `β = 20`, because on a Manhattan grid its cycles sum to exactly zero. The
  real lesson is that exploitability depends on your state space's geometry,
  which is precisely why you default to the potential-based form.
- **ε = 0.01 does not overtake ε = 0.1 within the usual 1000-step plot**
  (Lesson 1). It takes roughly 6000. The horizon you plot decides which
  algorithm appears to win.

## Where to go next

PPO, GAE, SAC for continuous actions, vectorised environments, and Atari with
a CNN. Lesson 5's README sketches each and where it follows from.

## Reference

Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) is the
standard text and follows roughly this order. Lessons 1–2 correspond to its
chapters 2 and 6; Lessons 3–4 to the function-approximation and policy-gradient
chapters.
