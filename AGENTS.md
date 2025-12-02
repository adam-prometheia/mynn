# Repository Guidelines

## Project Structure & Module Organization
Core functionality now lives inside the `mynn/` package. `mynn/neural_network.py` orchestrates training loops, metrics, and early stopping. `mynn/layers.py`, `mynn/batch_norm.py`, and `mynn/dropout.py` define trainable components, while `mynn/activations.py` and `mynn/losses.py` bundle forward/backward math for nonlinearities and criteria. Optimisers reside in `mynn/optimisers.py`, and data helpers (e.g., stratified splitting) live in `mynn/train_test_split.py`. Keep new layers or utilities in their existing topical module to preserve this focused layout; only add new modules when an existing one would otherwise exceed ~400 lines.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment before installing dependencies.
- `python -m pip install -r requirements-dev.txt`: install runtime + tooling dependencies and link the package via `-e .`.
- `python - <<'PY' ... PY`: run quick experiments inline; import from `mynn import Neural_Network`, wire layers, then feed numpy arrays to `train()`.
- `python scripts/<experiment>.py`: when an experiment grows beyond a snippet, add a file under `scripts/` so reviewers can re-run it verbatim.

## Coding Style & Naming Conventions
Use Black-compatible 4-space indentation, snake_case identifiers, and CapWords for public classes (e.g., `Layer_Dense`). Keep numpy operations vectorised and document math-heavy sections inline. Adopt the repo-wide 100-character line length configured for both Black and Ruff. Run `ruff check` before `black` to catch import order fixes, and mention any intentional deviations in the PR.

## Testing Guidelines
Author behavioural tests with `pytest` under `tests/`, mirroring the module path (`tests/test_layers.py`, etc.). Name tests after the scenario, e.g., `test_layer_dense_backward_shapes`. Aim for coverage on gradient flow, numerical stability, and failure modes (invalid shapes, dtype mismatches). Run `python -m pytest -q` locally and attach the command output to the PR description when touching critical math.

## Commit & Pull Request Guidelines
The repo lacks historical commits, so adopt a concise, imperative style such as `Add Adam weight decay option`. Squash noisy WIP commits before opening a PR. PRs must describe the motivation, list functional changes, call out new dependencies, and attach screenshots or loss/accuracy plots when training behaviour changes. Link to any upstream issue IDs and mention reviewers who should sanity-check the math.

## Agent-Specific Tips
Seed numpy via `np.random.seed(42)` inside experiments to produce reproducible traces when sharing plots. When inspecting gradients, stash interim arrays under `/tmp` or within `.gitignore`d notebooks to keep the repo clean.
