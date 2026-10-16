# Lesson 0 — The Gymnasium API

**Goal:** understand the agent–environment loop as Gymnasium expresses it, and  
build the `rollout` helper that every later lesson depends on.

There is no learning in this lesson. That is on purpose: before you can write  
an algorithm that improves a policy, you need a policy, an environment, and a  
way to measure. This lesson builds those three things.

## CartPole, the running example

Nearly every example below uses **CartPole**, so it is worth two minutes up  
front. It is the "hello world" of RL — small enough to train in seconds on a  
laptop, hard enough to be a real control problem.

A pole is hinged to a cart that slides along a frictionless track. The pole  
starts almost upright and gravity immediately begins tipping it over. You  
cannot touch the pole; you can only shove the cart left or right, and the  
pole's balance is a side effect of how you move the cart underneath it.

```
            │  ← pole, hinged at the cart, free to rotate
            │
        ┌───┴───┐
     ←──┤ cart  ├──→      your only choice: push left, or push right
        └──O─O──┘
    ════════════════════   track
```

It is the same problem as balancing a broom upright on your palm: the broom is  
the pole, your hand is the cart, and you keep it up by sliding your hand  
around under it.

Each step you observe four numbers and push once:

|   | Observation (4 numbers) | Actions (2) |
| --- | --- | --- |
|   | cart position (`x`) | `0` — push left |
|   | cart velocity (`ẋ`) | `1` — push right |
|   | pole angle (`θ`) |   |
|   | pole angular velocity (`θ̇`) |   |

The episode **ends** when the pole tips past ±12°, the cart runs off the track(±2.4 units), or you survive 500 steps. You score `+1` for every step survived,so the return is simply **how many steps you lasted**, and 500 is the maximum.CartPole is conventionally called "solved" at a mean return of 475 over 100consecutive episodes.

> **Where does 475 come from?** Convention, not derivation. It is 95% of the  
> 500-step cap — "almost always reach the ceiling," with slack so a few  
> unlucky starting states do not disqualify you. (`CartPole-v0` used 195 of  
> 200, the same idea.) The 100 episodes just averages out the randomised start.
> 
> These numbers date from the OpenAI Gym leaderboard era, when a round-number  
> target let submissions be labelled solved. It survives as  
> `env.spec.reward_threshold`, pure metadata — nothing in `env.step()` reads  
> it. Worth asking of any benchmark threshold: derived, or merely agreed upon?

Two properties make it a good teaching environment. The reward is _delayed_ —  
the push that dooms you happens long before the pole visibly falls — and the  
observation is four continuous numbers, so there is no finite list of states to  
tabulate. That second property is exactly the wall Lesson 3 has to break.

In Gymnasium it is `gym.make("CartPole-v1")`, and you will see it run at the  
end of this lesson.

## The formalism

RL problems are modelled as a **Markov Decision Process** (MDP), a 5-tuple  
`(S, A, P, R, γ)`:

| Symbol | Name | In Gymnasium |
| --- | --- | --- |
| `S` | state space | `env.observation_space` |
| `A` | action space | `env.action_space` |
| `P(s'\|s,a)` | transition dynamics | hidden inside `env.step`; you can sample it but not query it |
| `R(s,a,s')` | reward function | the `reward` returned by `env.step` |
| `γ` | discount factor | _yours_, not the environment's |

"5-tuple" just means the problem is fully specified once you have pinned down  
these five things. Taking each in turn, with CartPole as the running example.

### `S` — the state space

The set of every situation the agent can find itself in. A single member of  
that set is a **state**, written `s`.

CartPole's state is four numbers — cart position, cart velocity, pole angle,  
pole angular velocity — so `S` is the set of all possible combinations of  
those four. That is an infinite set, since the numbers are continuous. For  
FrozenLake in Lesson 2 it is just the 16 squares of the grid, which is finite  
and small enough to write in a table.

#### "Observation" — and why the code says `obs`, not `state`

One naming thing first, because the code uses a word the theory above does not.

An **observation** is what the environment actually shows you on a given step.A **state** is the environment's true internal condition. They are not alwaysthe same thing:

*   In **CartPole** they are identical. The environment's entire internalcondition _is_ those four numbers, and it hands you all four. State =observation.
*   In **poker** they differ sharply. The state includes everyone's hole cards;  
    your observation shows only your own. You are acting on a partial view.

Gymnasium cannot know which case you are in, so its API is honest and always  
says _observation_: `env.reset()` returns `obs`, and the space describing it is  
`observation_space`. `**obs**` **is simply the abbreviation of "observation"** that  
appears everywhere in RL code, including throughout this curriculum.

Because CartPole's state and observation coincide, this curriculum uses the two  
words interchangeably — and you can read `obs` as "the state" without losing  
anything. Where they genuinely differ you have a _partially observable_ problem  
(a POMDP), which needs machinery beyond these lessons. Lesson 3 brushes against  
it: a single Atari frame does not reveal which way the ball is moving, so DQN  
stacks four frames to build an observation worth treating as a state.

So, in the diagram below: `obs` is the four-number array, and for CartPole that  
array is the complete state.

#### What actually crosses the boundary

A state is not an abstraction in the code — it is a concrete array of a fixed  
shape and dtype. Here is one full step, with everything the environment hands  
you and everything you hand back:

```
                   +------------------------------+
                   |         ENVIRONMENT          |
                   |         CartPole-v1          |
                   +------------------------------+
                        |                     ^
     obs, info = reset()|                     | step(action)
                        v                     |
   obs  float32, shape (4,)                   action  int in {0, 1}
   +--------+--------+--------+--------+      +---------------+
   |  pos   |  vel   | angle  | angvel |      |    0 or 1     |
   +--------+--------+--------+--------+      +---------------+
      0.014   -0.023   -0.046   -0.048                ^
        x       x_dot    theta   theta_dot            |
                        |                             |
                        |      +----------------+     |
                        +----->|     AGENT      |-----+
                               | policy(obs)->a |
                               +----------------+

   step() returns a 5-tuple:
     next_obs     float32, shape (4,)   the new state
     reward       float                 +1.0
     terminated   bool                  pole fell / cart escaped
     truncated    bool                  hit the 500-step cap
     info         dict                  diagnostics, never for learning
```

In notation, writing `ℝ⁴` for "a vector of 4 real numbers":

```
S = ℝ⁴                  observation_space = Box(shape=(4,), dtype=float32)
A = {0, 1}              action_space      = Discrete(2)

policy:  ℝ⁴ → {0, 1}
```

#### What is `info`?

Metadata about the step: a dict for things useful to _you_ that the agent must  
never learn from. On CartPole it is literally empty, which makes it easy to  
overlook, but other environments put real diagnostics there:

| Environment | `info` after a step |
| --- | --- |
| CartPole | `{}` |
| FrozenLake (slippery) | `{'prob': 0.333}` — which branch of `P` you landed on |
| CliffWalking | `{'prob': 1.0}` — deterministic, so always 1 |
| wrapped in`RecordEpisodeStatistics` | `{'episode': {'r': 20.0, 'l': 20, 't': 0.0003}}` on the final step |

That FrozenLake `prob` is the environment telling you the probability of the  
transition that just occurred — a direct window onto the stochastic `P` from  
the section above.

**The rule:** `**info**` **is for logging, debugging and evaluation — never an input to**  
**the policy.** The observation is the agent's entire view of the world. Anything  
you pull out of `info` and feed to the agent is privileged information it will  
not have when you deploy it, and your Markov assumption no longer covers you.  
Lesson 5's GridWorld follows this: it reports `distance` and `agent` in `info`  
for the training script to print, while the agent sees only the integer  
observation.

(Strictly, `info` is a general out-of-band channel rather than guaranteed-inert  
decoration — some wrappers and vectorised setups use it to pass through data a  
correct implementation needs. But nothing in this curriculum does, so treat it  
as metadata throughout.)

#### From one state to a batch of them

Lesson 3 feeds these to a neural network, which never processes one state at a  
time — it processes a **batch**. That adds a leading dimension, and tracking it  
is most of the shape-debugging you will do:

| Stage | Shape | Meaning |
| --- | --- | --- |
| one observation | `(4,)` | a single state |
| batch of states | `(B, 4)` | `B` states stacked, e.g. `(64, 4)` |
| Q-network output | `(B, 2)` | one value per action, per state |
| action taken | `(B,)` | which action was actually chosen |
| gathered Q-value | `(B,)` | the value of\_that\_ action |

The network is `Linear(4 → … → 2)`: four numbers in, one number out peraction. Lesson 3's `gather` step — pulling `(B, 2)` down to `(B,)` using theactions actually taken — is exactly the move from "value of every action" to"value of the action we chose," and getting it wrong is a classic silent bug.

> **A detail that confuses everyone.** `observation_space` advertises bounds of  
> ±4.8 for position and ±0.419 rad (±24°) for angle — each exactly **twice**  
> the value at which the episode actually ends (±2.4 and ±12°). The space is  
> deliberately looser than the termination rule, so the state that trips  
> termination is still a legal member of the space. Velocities are unbounded  
> (`±inf`) because nothing caps them in principle. **The space says what is**  
> **representable, not what is survivable** — so never infer the goal from the  
> bounds.

Whether `S` is small and countable or huge and continuous turns out to decide  
which algorithms are even available to you. FrozenLake's 16 states fit in a  
table you could print; CartPole's `ℝ⁴` has uncountably many states and you will  
never see the same one twice, so there is nothing to tabulate. That is the wall  
Lesson 3 breaks through — and the reason it replaces the table with a function.

### `A` — the action space

The set of moves available to the agent. A single move is an **action**,  
written `a`.

CartPole has two: push the cart left, or push it right. Gymnasium calls that  
`Discrete(2)`, meaning the valid actions are the integers `0` and `1`. Every  
environment in this curriculum has a discrete action space. Continuous action  
spaces — "apply 0.37 newtons" — exist and need a different algorithm family.

### `P(s'|s,a)` — the transition dynamics

The rule governing what happens next. Read the notation `P(s'|s,a)` as "the  
probability of landing in state `s'`, **given** that you were in state `s` and  
took action `a`." The vertical bar means "given"; the apostrophe in `s'`  
(spoken "s prime") conventionally marks the _next_ state.

**In CartPole,** `**P**` **is deterministic.** It is just Newtonian physics: from a  
given configuration, pushing left produces exactly one next configuration. So  
for any `(s, a)` the distribution is degenerate — probability `1` on a single  
successor and `0` on every other state. Run the same `(s, a)` two hundred times  
and you get the identical `s'` every time. The only randomness in a CartPole  
episode is in `reset()`, which jitters the starting state; after that the  
trajectory is fully determined by the actions you choose.

Many environments are _not_ like this. On slippery FrozenLake, "move right"  
from the start square lands you on three different squares with probability  
exactly `1/3` each — intended direction, or either perpendicular one. There,  
one `(s, a)` genuinely has several possible successors.

`P` is written as a probability distribution to cover both cases in one  
formalism. Determinism is not a different kind of dynamics; it is the special  
case where all the probability mass sits on one outcome. That matters because  
the algorithms in this curriculum never assume determinism — they learn from  
sampled transitions, which works whether `P` is a coin flip or a certainty.  
(Lesson 2 trains the same agent on both the slippery and non-slippery  
FrozenLake to make the point concrete.)

**The dynamics are hidden from you.** `P` exists but you cannot query it. This  
is what makes the problem _reinforcement_ learning rather than planning — you  
learn from sampled experience, not from a model you were handed. (Lesson 5 is  
where you sit on the other side of this wall and write `P` yourself.)

### The "Markov" in Markov Decision Process

This is where the name comes from. The **Markov property** is the assumption  
that the current state contains everything relevant about the history — that  
the next state depends on `s_t` and `a_t` alone, and knowing the states before  
it would tell you nothing extra:

```
P(s_{t+1} | s_t, a_t)   is the same whether or not you also knew  s_{t-1}, s_{t-2}, …
```

CartPole satisfies this because the observation includes _velocities_. Position  
and angle alone would not be enough — a pole at 5° falling fast needs a  
different response from one at 5° and steady — but once velocity is in the  
state, the past adds nothing.

The Atari example from earlier is the same point: a single frame fails this  
test, so stacking four of them is what makes the observation Markov enough to  
learn from.

The assumption matters because every algorithm here relies on it. They all  
learn a value or policy as a function of the current state only, so if the  
state omits something decisive, no amount of training recovers it. When an  
agent mysteriously plateaus, "is my state actually Markov?" is a good early  
question.

### `R(s,a,s')` — the reward function

The environment's feedback signal: a **single number** returned after every  
step, saying how good that transition was. It is the only thing telling the  
agent what you actually want, which makes it the most important design  
decision in any RL problem.

The notation `R(s,a,s')` means the reward can depend on where you were, what  
you did, and where you ended up. Most environments use something far simpler:

| Environment | Reward |
| --- | --- |
| CartPole | `+1` for every step the pole stays up |
| FrozenLake | `+1` for reaching the goal, `0` everywhere else |
| CliffWalking | `−1` per step, `−100` for falling off the cliff |
| Lesson 5's GridWorld | `−1` per step, `−2` into a wall, `+10` at the goal |

Three things worth absorbing now, because they recur in every later lesson:

**The reward encodes the goal, not the method.** CartPole pays `+1` per step  
and says nothing about pole angles or cart velocity. "Keep the pole up as long  
as possible" is the entire specification; _how_ is the agent's problem. The  
discipline here is to reward **what you want**, never what you imagine the  
strategy should be — specifying the method in the reward is how you get an  
agent that does exactly what you asked and not at all what you meant.

**A negative reward per step is how you ask for speed.** CliffWalking's `−1`  
per step means dawdling costs you, so the agent prefers short paths. Without  
it, a policy wandering for 500 steps would score the same as the direct route.

**Rewards arrive later than the actions that earned them.** On CartPole every  
step pays the same `+1`, so the reward you collect at step 40 tells you nothing  
about whether step 40 was a good move — it is the payoff for staying balanced,  
which was secured by a well-chosen push twenty steps earlier. A single reward  
is rarely attributable to the action it arrives with.

CartPole makes this vivid: it pays `+1` on _every_ step, including the final  
one where the pole topples. There is no penalty anywhere in the reward — the  
only signal that things went badly is that the rewards **stopped coming**.  
Good play is therefore measured purely by how many `+1`s you collect, which is  
why the return is just the episode length.

So the agent must work out which earlier actions earned the reward that showed  
up later. That is the **credit assignment problem**, and essentially every  
algorithm in this curriculum is a different answer to it. The return `G_t` in  
the next section is the first move toward solving it: by scoring each action  
with the total reward that _followed_ it, rather than just the reward that  
accompanied it, an action starts getting credit for the future it bought.

### `γ` — the discount factor

**γ belongs to you, not the environment.** The discount factor is part of your  
_objective_, not part of the problem specification: it sets how much you care  
about reward now versus reward later. Before that claim means anything,  
though, we need the notation it rests on.

#### Reading the notation

An episode is a sequence of timesteps. `t` is just a counter over them —  
`t = 0` is the first step, `t = 1` the second, and so on. At each step the  
agent sees a state, picks an action, and gets a reward back:

```
t:        0         1         2         3
state:    s₀   →    s₁   →    s₂   →    s₃  ...
action:   a₀        a₁        a₂        a₃
reward:      r₁        r₂        r₃
```

A subscript is "the value of this thing at that timestep": `s₂` is the state  
at step 2, `a₂` the action taken there. Rewards are indexed one ahead by  
convention — `r₁` is the reward that arrives _after_ acting in `s₀` — which is  
why the sums below start at `r_{t+1}` rather than `r_t`. That off-by-one trips  
up everyone the first time; it is purely bookkeeping, nothing conceptual.

On CartPole every step that keeps the pole up pays `r = 1`, so a 12-step  
episode produces the reward sequence `1, 1, 1, …, 1` (twelve of them).

#### What `G_t` is

`G` stands for **return** (the `G` is conventional, from "gain"). `G_t` is the  
total reward collected **from step** `**t**` **onward to the end of the episode** —  
not the reward at step `t`, and not the total for the whole episode unless  
`t = 0`.

For that 12-step CartPole episode, counting every reward equally:

*   `G_0 = 12` — twelve steps remain from the very start
*   `G_5 = 7` — only seven steps remain from step 5
*   `G_11 = 1` — one final step remains

So `G_t` answers "**how much reward follows from here?**" The point of  
computing it per-step is credit assignment: each action gets scored by what  
came after it, not by the episode as a whole.

#### Adding the discount

Now reward the near future more than the distant future. Multiply each reward  
by `γ` raised to how many steps away it is:

```
G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + γ³·r_{t+4} + ...
```

The next reward is undiscounted (`γ⁰ = 1`), the one after is scaled by `γ`,  
the one after that by `γ²`, and so on — each step further away is worth  
geometrically less. With `γ = 0.9`, a reward 10 steps out counts for  
`0.9¹⁰ ≈ 0.35` of its face value, and at 50 steps barely 0.005.

The compact form is the same statement with a summation sign:

```
G_t = Σ_{k=0}^∞ γ^k · r_{t+k+1}
```

Read `Σ` (sigma) as "add up the following for every value of `k`". `k` counts  
how many steps ahead we are looking: `k = 0` gives the `γ⁰·r_{t+1}` term,  
`k = 1` gives `γ¹·r_{t+2}`, and so on. It is the expanded line above, written  
once instead of spelled out term by term.

#### Why discount at all

Two reasons, one mathematical and one practical.

**It makes the sum finite.** For a task that never terminates, adding up  
infinitely many rewards of 1 gives infinity, and "maximise infinity" is not an  
objective you can optimise. With `γ < 1` the sum converges to at most  
`1/(1−γ)` — for `γ = 0.99` that ceiling is 100, which is exactly the number  
Exercise 1 asks you to observe.

**It encodes how far ahead you care.** `γ = 0` makes the agent purely  
myopic — it optimises only the very next reward and ignores everything after.  
`γ → 1` makes it far-sighted, but every action's score now depends on hundreds  
of subsequent random events, so the estimates become noisy and learning slows.  
`γ = 0.99` is the usual compromise and is what this curriculum uses throughout.

A useful intuition: `1/(1−γ)` is roughly the agent's planning horizon in  
steps. `γ = 0.9` ≈ 10 steps of foresight; `γ = 0.99` ≈ 100.

This is also why γ is _yours_: the environment never mentions it. Two people  
can face the identical CartPole and optimise genuinely different objectives  
by choosing different γ.

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

|   | Meaning | Value of the final state | Bootstrap? |
| --- | --- | --- | --- |
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
env.observation_space       # Box(-inf, inf, (4,), float32) for CartPole
env.action_space            # Discrete(2)
env.action_space.sample()   # -> 0 or 1, drawn uniformly at random
```

`Discrete(n)` means actions are integers `0..n-1`. `Box` is a continuous vector  
with per-dimension bounds. Lessons 1–5 all use `Discrete` action spaces; a  
`Box` _action_ space would require a different family of algorithms (DDPG, SAC,  
PPO with a Gaussian head) which this curriculum does not cover.

### What `.sample()` does

Every space can draw a **random element of itself**. For CartPole's  
`Discrete(2)` that is "pick 0 or 1, each with probability ½" — over 10,000  
draws you get almost exactly 50/50.

Its main use is writing a random policy without hardcoding the valid actions:

```python
def policy(obs):
    return int(env.action_space.sample())
```

That is the random baseline this lesson benchmarks against, and it works  
unchanged on CartPole's 2 actions or Taxi's 6, because the space knows its own  
range.

Two companions you will meet in later lessons:

*   `**space.seed(0)**` makes the draws reproducible — same seed, same sequence.
*   `**space.contains(x)**` asks whether `x` is a legal member. `contains(0)` is  
    `True` for `Discrete(2)`; `contains(5)` is `False`. Lesson 5's GridWorld uses  
    this to reject invalid actions.

> `**observation_space.sample()**` **is rarely what you want.** It draws  
> **uniformly from the box** — and the box, as noted above, is twice as wide as  
> the termination limits on both bounded dimensions:
> 
> |   | box | terminates past | fraction inside |
> | --- | --- | --- | --- |
> | `x` | ±4.8 | ±2.4 | ½ |
> | `θ` | ±0.419 | ±0.209 | ½ |
> 
> So only `½ × ½ = ¼` of draws are even non-terminal (measured: 0.2497 over  
> 200k samples). The other three-quarters are states where the episode would  
> already have ended. The velocities never constrain this — they are unbounded,  
> so termination never depends on them.
> 
> The remaining quarter are legal states, but still a distribution unlike  
> anything a policy visits: sampling spans cart velocities of roughly ±4, while  
> real play stays within about ±1.9.
> 
> The underlying point: `observation_space` is **not** the state space `S`. It  
> is a loose rectangular _container_ around it, chosen so every observation the  
> env emits fits inside. `S` itself — the states actually reachable by the  
> physics — is a thin, curved subset, and the `Box` has no way to describe it.
> 
> So it is useful for checking that a network accepts the right shape, and a  
> poor substitute for real experience — anything fit on uniform samples is  
> being trained mostly on states it will never encounter.
> 
> Note also that `sample()` returns a **numpy** integer, not a Python `int`,  
> which is why the code here wraps it in `int(...)`. A numpy integer can fail a  
> `contains` check in some Gymnasium versions — an irritating bug to chase.

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

Both live in [`exercises.py`](exercises.py). They raise `NotImplementedError`until you fill them in.

1.  `**episode_return(env, policy, gamma)**` — run one episode and return the\_discounted\_ return. Verify that with `gamma=1.0` it matches the undiscountedsum, and that with `gamma=0.99` on CartPole it saturates near `1/(1-γ) = 100`no matter how long the pole stays up.
2.  `**ObservationScaler(env, low, high)**` — a wrapper mapping each observation  
    dimension linearly from `[low, high]` into `[-1, 1]`. You must update  
    `observation_space`, not just the values: downstream code reads the space to  
    size its networks. This is the preprocessing DQN needs when observation  
    dimensions have wildly different magnitudes.

Check yourself:

```
uv run pytest tests/test_lesson0.py
```

Compare against [`solution/exercises.py`](solution/exercises.py) once you have  
something working — not before.

## Run the demo

```
uv run python -m rl_tutorial.lesson0_api.demo
```

This compares a random policy against a hand-coded heuristic on CartPole. The  
heuristic — "push in the direction the pole is already falling", reacting to  
angular _velocity_ rather than angle — scores around 200 versus random's ~23,  
without a single gradient step.

That number is worth sitting with. A one-line controller with no parameters  
gets most of the way to solving CartPole. When your DQN in Lesson 3 hits 150  
and you feel good about it, remember this line of code beat it. **Always**  
**benchmark against both random and the best trivial policy you can think of** —  
a surprising amount of published RL has been embarrassed by exactly this.

## Key takeaways

*   `terminated` and `truncated` are different and the difference matters in everyupdate rule you will write.
*   γ is part of your objective, not the environment.
*   Evaluate over _many_ episodes with a fixed seed offset; a single episode'sreturn is nearly meaningless.
*   Always benchmark against a random policy.