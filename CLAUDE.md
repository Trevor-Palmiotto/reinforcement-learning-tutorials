# CLAUDE.md

A six-lesson, learn-by-building reinforcement learning curriculum on
Gymnasium. See `README.md` for the course itself.

## Layout and commands

- Package: `rl_tutorial/` — one subpackage per lesson (`lesson0_api` …
  `lesson5_custom_env`), plus `common/` for shared rollout, schedule and
  plotting helpers.
- Each lesson has a `README.md`, an `exercises.py` of stubs, a `solution/`
  reference, a runnable experiment script, and `tests/test_lessonN.py`.
- uv project, Python ≥ 3.12. Setup: `uv sync`.
- Fast tests (~15 s): `uv run pytest tests -m "not slow"`.
  Full suite includes end-to-end training and takes minutes.
- Tests against an unimplemented stub **skip** (via `_skip_if_stub`), they do
  not fail. A clean run with many skips means "not done", not "passing".

## Two branches: `main` and `class`

- **`main`** is the clean, teachable course. Every exercise in `exercises.py`
  keeps its `raise NotImplementedError(...)` stub.
- **`class`** is the owner's working branch: everything on `main` **plus** their
  filled-in exercise answers.

Rules:

- Lesson improvements (README explanations, docstrings, test failure
  messages, experiments, shared code) belong on **both** branches.
- Answers belong **only** on `class`. Never let a worked answer reach `main`,
  and never fill in a stub when working on `main`.
- Check which branch you are on before editing an `exercises.py`.

Porting workflow (make the change on `main`, then merge into `class`):

1. Commit the improvement on `main`. If it was first made on `class`, copy the
   files across with `git checkout class -- <files>`, then restore the
   `NotImplementedError` stubs in any touched `exercises.py` by hand and drop
   imports that only the answer used.
2. `git checkout class && git merge main` so `class` stays a superset.

Leak check: `git diff main class` should touch only `exercises.py` files and
show nothing but answer bodies. Comparing `pytest -rA` PASSED lists between the
branches is the stronger check: every extra pass on `class` should be an
exercise test for a lesson the owner has answered.

If you can only push to a side branch (e.g. a cloud session), say in the PR
description which of `main` / `class` the change is for, and whether it still
needs porting to the other.

## Writing standards for lesson material

- **Run it before you claim it.** Do not write a numeric result or behavioural
  claim into a README from memory or from the textbook. Run the experiment and
  write what actually happened. When a measurement contradicts the expected
  result, keep the contradiction and explain it — it is usually the better
  lesson.
- **Define notation before use.** Every symbol, subscript and environment
  (e.g. CartPole) is introduced before it appears in an equation or example.
- **Treat reader confusion as a defect in the writing**, not in the reader.
- **Keep explanations proportionate.** After expanding one, check whether it
  has outgrown the question it answers.
- Test failure messages should be diagnostic: say what was expected, what was
  got, and the likely mistake.
