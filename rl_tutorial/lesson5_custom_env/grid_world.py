"""Lesson 5 exercises -- fill in the TODOs.

    uv run pytest tests/test_lesson5.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

#: Action index -> (row delta, col delta). Up, Right, Down, Left.
MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
#: Arrow per action, for rendering the greedy policy (matches MOVES order).
ARROWS = "^>v<"


class GridWorldEnv(gym.Env):
    """A square grid. Start top-left, reach the goal bottom-right.

    Rewards:
        -1 per step (this is what makes short paths preferable)
        +10 on reaching the goal
        -1 extra for walking into a wall (the move is refused)

    Observations are a single integer ``row * size + col``, i.e. a
    ``Discrete(size * size)`` space. That is deliberate: it lets the tabular
    Q-learning agent from Lesson 2 index it directly, with no adapter.

    Args:
        size: Grid edge length.
        obstacles: Optional ``(row, col)`` cells that cannot be entered.
        render_mode: ``"ansi"``, ``"rgb_array"``, or None.

    You must implement ``__init__``, ``reset``, ``step``, and ``_get_obs``.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        *,
        obstacles: tuple[tuple[int, int], ...] = (),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        # TODO: store size, obstacles (as a set), render_mode; set
        #       self.observation_space = Discrete(size * size)
        #       self.action_space = Discrete(4)
        #       self.start = (0, 0); self.goal = (size - 1, size - 1)
        raise NotImplementedError("Exercise 1: implement GridWorldEnv.__init__")

    def _get_obs(self) -> int:
        """Encode ``self.agent`` as a single Discrete index.

            obs = row * size + col

        Return a plain Python ``int``. A numpy integer will fail
        ``observation_space.contains`` in some Gymnasium versions, and
        ``check_env`` will tell you so.
        """
        raise NotImplementedError("Exercise 4: implement _get_obs")

    def _get_info(self) -> dict:
        """Diagnostics. Provided for you.

        ``info`` is the right place for anything the agent should NOT learn
        from but you want to log. Putting privileged state here is fine;
        putting it in the observation would be cheating.
        """
        return {
            "distance": self._manhattan(self.agent, self.goal),
            "agent": self.agent,
        }

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        MUST call ``super().reset(seed=seed)`` FIRST -- that is what seeds
        ``self.np_random``. Everything random in this class must draw from
        ``self.np_random``, never the global ``np.random``.

        Supports ``options={"random_start": True}`` to place the agent
        uniformly at random (but never on the goal or an obstacle).

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)
        # TODO: set self.agent, then return (self._get_obs(), self._get_info())
        raise NotImplementedError("Exercise 2: implement reset")

    def step(self, action: int):
        """Apply ``action``.

        Rules:
            - Moving off the grid or into an obstacle: the agent does not move
              and the reward is -2 (the usual -1 plus a -1 wall penalty).
            - Reaching the goal: reward +10 and ``terminated=True``.
            - Otherwise: reward -1.

        ``truncated`` is always False here. The step limit is applied by the
        ``TimeLimit`` wrapper that ``gym.make`` adds -- that is the wrapper's
        job, not the environment's, and doing it yourself double-counts.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        raise NotImplementedError("Exercise 3: implement step")

    def render(self):
        """Render per ``self.render_mode``. Provided for you."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        rows = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.agent:
                    row += "A"
                elif (r, c) == self.goal:
                    row += "G"
                elif (r, c) in self.obstacles:
                    row += "#"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def _render_rgb(self) -> np.ndarray:
        """An (H, W, 3) uint8 image, 32 pixels per cell."""
        cell = 32
        img = np.full((self.size * cell, self.size * cell, 3), 250, dtype=np.uint8)

        def fill(pos, color):
            r, c = pos
            img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = color

        for obstacle in self.obstacles:
            fill(obstacle, (60, 60, 60))
        fill(self.goal, (27, 175, 122))    # aqua, matching the chart palette
        fill(self.agent, (42, 120, 214))   # blue
        img[::cell, :, :] = 200
        img[:, ::cell, :] = 200
        return img


def potential_shaping(
    state: tuple[int, int],
    next_state: tuple[int, int],
    goal: tuple[int, int],
    gamma: float,
) -> float:
    """Potential-based shaping term, which cannot change the optimal policy.

        F(s, s') = gamma * Phi(s') - Phi(s)      with  Phi(s) = -distance(s, goal)

    Ng et al. (1999) proved this form -- and only this form -- leaves the
    optimal policy unchanged for ANY environment, while still densifying the
    reward.

    The naive alternative ``distance(s) - distance(s')`` is the same expression
    with ``gamma = 1``. On THIS grid the two behave equivalently and both find
    the optimal policy, because distance changes symmetrically by +-1 so every
    cycle of naive shaping sums to exactly zero. That is a property of the grid
    geometry, not of the naive formula -- change the state space and the
    guarantee evaporates, which is precisely why the potential-based form is
    the one to default to. ``train.py --shaping`` measures both, and the README
    works through the cycle sums.

    Returns:
        The shaping term to ADD to the environment reward.
    """
    raise NotImplementedError("Exercise 5: implement potential_shaping")
