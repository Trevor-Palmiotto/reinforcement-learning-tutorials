# Lesson 5 — Build Your Own Environment

**Goal:** sit on the other side of the interface. Every lesson so far consumed
`env.step`; now you write it.

This closes the loop in a satisfying way: at the end you train **Lesson 2's
Q-learning agent, completely unmodified**, on an environment you wrote from
scratch. If the abstraction holds, that just works — and that is the real test
of whether you implemented `Env` correctly.

## The contract

Subclassing `gymnasium.Env` means promising six things:

| Member | Promise |
|---|---|
| `observation_space` | A `Space`. Every observation you emit must be `in` it. |
| `action_space` | A `Space`. Every action you accept must be `in` it. |
| `reset(seed, options)` | Returns `(obs, info)`. Starts a fresh episode. |
| `step(action)` | Returns `(obs, reward, terminated, truncated, info)`. |
| `render()` | Visualises per `render_mode`. Optional. |
| `metadata` | Declares `render_modes` and `render_fps`. |

Four places where this goes wrong in practice:

**1. Seeding.** `reset` must call `super().reset(seed=seed)` **first**. That
initialises `self.np_random`, the per-instance RNG. Use it for every random
draw — never the global `np.random`, which makes runs irreproducible and makes
parallel environments share a stream.

**2. Observation dtype.** If your space is `Box(dtype=np.float32)` you must
emit `float32`. NumPy loves promoting things to `float64` and `check_env` will
catch it, but only if you run it.

**3. `terminated` vs `truncated`.** Same distinction as always, now yours to
get right. Reaching the goal is `terminated`. Hitting a step limit is
`truncated`. If you conflate them, every algorithm in this curriculum learns
the wrong value function from your environment — and the bug looks like an
agent bug, not an environment bug.

**4. Copying observations.** If you return a reference to a mutable internal
array, the agent's replay buffer fills with aliases that all change together.
Return a copy.

## Reward design

Our grid gives `-1` per step and `+10` at the goal. The `-1` is what makes the
agent prefer short paths; without it, a policy that wanders for 500 steps
before arriving scores the same as the direct route.

**Reward shaping** adds a term to speed up learning — typically a nudge toward
the goal:

```
shaped_reward = r + β·(distance_before − distance_after)      # "naive"
```

This helps when rewards are sparse. It is also a well-known way to break an
environment, because a badly-chosen shaping term can change *which policy is
optimal* rather than merely accelerating the search for it. The classic
cautionary tale is an agent that learns to oscillate — step toward the goal,
step back, repeat — farming an approach bonus forever and never finishing.

Ng et al. (1999) proved when this is safe. Shaping is **policy-invariant** if
and only if it is *potential-based*:

```
F(s, s') = γ·Φ(s') − Φ(s)
```

for some potential `Φ` over states. With `Φ(s) = −distance(s, goal)` the
optimal policy provably cannot change.

### What actually happens on this grid — and why

Run `train.py --shaping` and you will find **all three variants reach the
optimal 10-step policy**, including the naive one, and they still do at
`β = 20`. The cautionary tale does not reproduce here. It is worth
understanding why, because the reason is more useful than the warning.

Compute the shaping collected over a two-step cycle — move one step closer to
the goal, then move back:

| | toward goal | back | **cycle sum** |
|---|---|---|---|
| naive | `+1.00` | `−1.00` | **`0.00`** |
| potential-based (γ=0.99) | `+1.05` | `−0.94` | **`+0.11`** |

On a Manhattan grid with `Φ = −distance`, the naive form is
`Φ(s) − Φ(s')` — exactly the potential-based form with `γ = 1`. Distance
changes by ±1 symmetrically, so **every cycle sums to exactly zero** and there
is nothing to farm. The naive version is accidentally safe on *this* geometry.

The mild irony: it is the *potential-based* term that pays `+0.11` per cycle
here, because `γ < 1` discounts the return leg less than the outbound one.
That does not violate the theorem — the environment still charges `−2` for the
two steps, so loitering nets `−1.89` and never beats finishing. Ng et al.
guarantee the optimal policy is preserved, not that every cycle sums to zero.

The real lesson is therefore sharper than "naive shaping is dangerous":

- Whether a shaping term is exploitable depends on the **geometry of your
  state space**, not just the formula. Reasoning about it in the abstract will
  mislead you.
- Potential-based shaping is the form that is safe **regardless** of geometry,
  which is exactly why you should default to it — not because the alternative
  always fails, but because you cannot tell in advance when it will.
- Always evaluate the learned policy on the **unshaped** environment. That is
  the only number that means anything, and `train.py --shaping` does it.

## Registration

Registering makes your env a first-class citizen:

```python
gym.register(id="tutorial/GridWorld-v0", entry_point=...)
env = gym.make("tutorial/GridWorld-v0", size=8)
```

`gym.make` also applies `TimeLimit` (via `max_episode_steps`) and the order
enforcement wrapper, which is why you should use it rather than constructing
the class directly.

## Your exercises

In [`grid_world.py`](grid_world.py):

1. **`GridWorldEnv.__init__`** — spaces and configuration.
2. **`reset`** — seed correctly, place the agent, return `(obs, info)`.
3. **`step`** — move, compute reward, decide termination.
4. **`_get_obs`** — encode `(row, col)` as a single `Discrete` integer, so that
   the tabular agent from Lesson 2 can index it directly.
5. **`potential_shaping`** — the policy-invariant shaping term.

```bash
uv run pytest tests/test_lesson5.py
```

## Verify and run

```bash
# Gymnasium's own conformance checker -- must pass clean
uv run python -m rl_tutorial.lesson5_custom_env.check

# Train Lesson 2's Q-learning agent on YOUR environment, unmodified
uv run python -m rl_tutorial.lesson5_custom_env.train

# Compare no shaping / naive shaping / potential-based shaping
uv run python -m rl_tutorial.lesson5_custom_env.train --shaping
```

`check_env` is worth running every time you touch the environment. It verifies
the space/observation agreement, the reset signature, seeding determinism, and
a dozen other details that are tedious to check by hand and silently corrupt
learning when wrong.

## Key takeaways

- `Env` is a small contract, and most of the work is in honouring its details
  exactly: seeding through `self.np_random`, matching dtypes, copying
  observations, and keeping `terminated` distinct from `truncated`.
- Run `check_env`. It finds real bugs in minutes that would otherwise look like
  mysterious agent failures.
- Reward shaping is powerful and dangerous. Potential-based shaping is the form
  that is provably safe.
- A correct environment works with any correct agent. That is the whole point
  of the interface, and training Lesson 2's code on your grid proves it.

## Where to go next

You now have the core of modern RL. The natural continuations:

- **PPO** — the current default for most problems. Takes Lesson 4's policy
  gradient and adds a clipped objective so you can safely take several
  gradient steps per batch of data, recovering much of the sample efficiency
  that on-policy methods lose.
- **GAE** — the smooth interpolation across Lesson 4's `n` dial.
- **SAC / DDPG** — continuous action spaces, which none of these lessons cover.
- **Vectorised environments** — `gym.make_vec` runs many copies in parallel and
  is how any of this gets done at realistic scale.
- **Atari** — the same DQN from Lesson 3, plus a CNN and frame stacking (recall
  from Lesson 0 that a single frame is not Markov).
