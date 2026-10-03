"""Run Gymnasium's conformance checker against the custom environment.

    uv run python -m rl_tutorial.lesson5_custom_env.check
    uv run python -m rl_tutorial.lesson5_custom_env.check --mine

``check_env`` verifies things that are tedious to check by hand and that
silently corrupt learning when wrong: that observations are actually inside
``observation_space``, that ``reset`` accepts and honours ``seed``, that the
five-tuple has the right types, that the same seed reproduces the same episode.

Run it every time you change the environment. A bug here surfaces later as a
mysterious agent failure, and you will look for it in the wrong place.
"""

from __future__ import annotations

import argparse

import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_tutorial.lesson5_custom_env.registration import SOLUTION_ID, YOURS_ID, make


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine", action="store_true", help="check your grid_world.py")
    parser.add_argument("--size", type=int, default=5)
    args = parser.parse_args()

    env_id = YOURS_ID if args.mine else SOLUTION_ID
    print(f"Checking {env_id} (size={args.size})")
    print("=" * 64)

    env = make(env_id, size=args.size, render_mode="ansi")

    # skip_render_check=False also exercises render(); our ansi renderer is
    # cheap so there is no reason to skip it.
    check_env(env.unwrapped, skip_render_check=False)
    print("check_env: PASSED")

    obs, info = env.reset(seed=0)
    print(f"\nreset(seed=0) -> obs={obs}, info={info}")
    print(env.unwrapped.render())

    print("\nWalking right then down to the goal:")
    total = 0.0
    for action in [1] * (args.size - 1) + [2] * (args.size - 1):
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    print(f"  final obs={obs} return={total} terminated={terminated}")
    print(env.unwrapped.render())

    # Determinism: the same seed must reproduce the same episode exactly.
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert a == b, "same seed produced different states"
    print("\nSeeding determinism: PASSED")

    rgb = make(env_id, size=args.size, render_mode="rgb_array")
    rgb.reset(seed=0)
    frame = rgb.unwrapped.render()
    assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
    print(f"rgb_array render: PASSED (shape {frame.shape}, dtype {frame.dtype})")

    env.close()
    rgb.close()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
