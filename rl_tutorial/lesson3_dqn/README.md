# Lesson 3 — Deep Q-Networks

**Goal:** replace the Q-table with a neural network, and learn why that one
change breaks everything until you add two specific fixes.

## Why the table has to go

Lesson 2's table has one entry per state. CartPole's state is four real
numbers — uncountably many states, none of which you will ever see twice. The
table cannot even be allocated, let alone filled.

So approximate it. Let a network `Q(s,a; θ)` with parameters `θ` take the state
and output one value per action. Now similar states share parameters and
experience **generalises**: learning about one pole angle teaches you about
nearby angles. This is the entire idea of deep RL.

The loss is the TD error from Lesson 2, squared:

```
L(θ) = E_(s,a,r,s')~D [ ( r + γ·(1−terminated)·max_a' Q(s',a'; θ⁻) − Q(s,a; θ) )² ]
                          └──────────── the target, y ────────────┘
```

Compare to the tabular update `Q(s,a) ← Q(s,a) + α(y − Q(s,a))`: that *was* a
gradient step on this loss, for a table. Same algorithm, different function
class.

## The two fixes, and why naive DQN fails without them

Swap a table for a network naively and training diverges. Two reasons, two
fixes — and the ablation script lets you watch both failures happen.

### 1. Experience replay

Consecutive transitions in an episode are heavily correlated: pole slightly
left, pole slightly left, pole slightly left. SGD assumes i.i.d. samples;
feeding it correlated batches means each gradient step is computed on a narrow
slice of state space and promptly undone by the next one. Worse, the data
distribution shifts as the policy changes — the network forgets states it no
longer visits (**catastrophic forgetting**).

Fix: store transitions in a large buffer and sample **uniformly at random**.
This decorrelates the batch and lets each transition be reused many times,
which also makes DQN far more sample-efficient than it would otherwise be.

### 2. The target network

Look at the loss again. The target `y` depends on `θ`. So every gradient step
moves the prediction *and* the target it is being fit to — chasing your own
tail. In practice the values spiral upward and diverge.

Fix: compute the target with a **frozen copy** `θ⁻`, synced from `θ` every
`target_sync` steps. Between syncs the target is a fixed function, so each
interval is ordinary supervised regression against a stable objective.

`(1 − terminated)` masks the bootstrap, still keyed on `terminated` and never
on `truncated` — the same rule from Lesson 0, now with real consequences: mask
on truncation here and CartPole plateaus, because you teach the agent that
surviving to the 500-step limit is worth nothing.

## Double DQN

Standard DQN's target uses `max_a' Q(s', a'; θ⁻)`. That single `max` both
**selects** the best action and **evaluates** it, using the same noisy
estimates. Since `E[max(noise)] ≥ max(E[noise])`, this systematically
*overestimates* — the max picks whichever action happens to have the luckiest
error, and then trusts that lucky number.

Double DQN decouples the two:

```
DQN:        y = r + γ · Q(s', argmax_a' Q(s',a'; θ⁻); θ⁻)
Double DQN: y = r + γ · Q(s', argmax_a' Q(s',a'; θ ); θ⁻)
                            └─ online net selects ─┘  └ target net evaluates ┘
```

An action must be *both* lucky in the online net and lucky in the target net to
be overvalued, which is much rarer. It is a two-line change and you can measure
the effect: the ablation prints mean predicted Q against the true discounted
return.

## Your exercises

In [`exercises.py`](exercises.py):

1. **`ReplayBuffer`** — a fixed-capacity numpy ring buffer. Preallocate arrays
   and overwrite the oldest entry; do not use a list of tuples, which allocates
   per step and is slow.
2. **`QNetwork`** — an MLP mapping observation to one Q-value per action. Note
   the output layer has **no activation**: Q-values are unbounded returns, not
   probabilities. Putting a ReLU or tanh here is a classic and silent bug.
3. **`compute_td_loss`** — build the target under `torch.no_grad()`, gather the
   predicted `Q(s,a)`, and return the loss. Supports both DQN and Double DQN.

```bash
uv run pytest tests/test_lesson3.py
```

### Two details worth care

**`gather` vs `max`.** The network outputs `(batch, n_actions)`. You need the
value of the action *actually taken*: `q.gather(1, actions.unsqueeze(1))`, not
a max. Getting this wrong trains the network toward its own maximum regardless
of behaviour and it will not learn.

**`no_grad` on the target.** The target is a constant label. If gradients flow
into it you are optimising the target to match the prediction as much as the
reverse, and it will not converge. Wrap target computation in
`with torch.no_grad():`.

**Huber loss over MSE.** A large TD error squared gives an enormous gradient;
`smooth_l1_loss` is quadratic near zero and linear in the tails, so one
outlying transition cannot blow up the weights.

## Run it

```bash
# Train DQN on CartPole (~2-4 minutes on CPU)
uv run python -m rl_tutorial.lesson3_dqn.train

# Double DQN
uv run python -m rl_tutorial.lesson3_dqn.train --double

# Watch it break: no replay, and no target network
uv run python -m rl_tutorial.lesson3_dqn.ablate
```

TensorBoard logs go to `runs/`:

```bash
uv run tensorboard --logdir runs
```

## What to expect

CartPole-v1 is conventionally called solved at a mean return of 475 over 100
episodes — a round 95% of the step cap rather than a derived figure, as
Lesson 0 explains. With the
shipped hyperparameters the best checkpoint reaches **500 (the maximum) on
every seed tested**, for both DQN and Double DQN.

But watch the per-episode returns during training and you will see they are
**wildly non-monotonic** — climbing to 300, collapsing to 50, recovering. This
is normal. As ε decays the agent reaches states its network never trained on,
and bootstrapped targets amplify the resulting errors until replay averages
them back out.

That instability has a practical consequence the training loop handles for
you, and it is worth understanding rather than skipping. **The weights from the
final episode are not the best weights you trained.** Measured here while
building this lesson, with an identical configuration and only the seed
changed, the final-episode policy scored 9.3 on one seed and 401.9 on
another — pure luck about where in the oscillation training happened to stop.
`train_dqn` therefore evaluates the greedy policy every `eval_every` episodes
and returns the best checkpoint. Any serious RL code does this; it is not a
nicety.

Remember Lesson 0: the one-line heuristic scores ~200. A DQN that plateaus near
200 has learned something less impressive than it looks.

## Key takeaways

- Function approximation buys generalisation and costs stability.
- Replay decorrelates samples; the target network stops the target moving.
- The `max` in the target causes systematic overestimation; Double DQN fixes it
  cheaply.
- Always benchmark against both random and the trivial heuristic.
