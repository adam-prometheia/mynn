# Modular Neural Networks in NumPy

A lightweight, fully vectorised neural-network playground built purely on NumPy. The library exposes composable layers, activations, regularisers, optimisers, and utility helpers so you can prototype small models, inspect gradients, and iterate on training loops without heavy dependencies.

## Features
- Dense, dropout, and batch-normalisation layers with explicit forward/backward APIs.
- Softmax and ReLU-family activations plus numerically stable categorical cross-entropy loss.
- Adam optimiser with pluggable learning-rate parameters.
- Training harness featuring mini-batching, accuracy/loss tracking, and early stopping.
- Simple train/test split helper for quick experiments.

## Repository Layout
```
README.md            Project overview (this file)
AGENTS.md            Contributor/agent guidelines
mynn/                Source package (layers, activations, losses, optimiser, trainer)
scripts/             Reproducible experiments and demos
tests/               Pytest suite with gradient and behaviour checks
pyproject.toml       Tooling + packaging metadata
requirements-dev.txt Development dependencies (pytest, ruff, black)
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/spiral_demo.py            # 2-class spiral classification demo
# Extras: nnfs multi-class spiral demos (functional vs OOP style)
# python scripts/nnfs_spiral_functional.py
# python scripts/nnfs_spiral_oop.py
```

To run checks before a commit:
```bash
ruff check mynn
black mynn tests scripts
pytest -q
```

## Status & Roadmap
The codebase is pre-1.0 and optimised for learning and experimentation. Planned improvements include additional optimisers, convolutional layers, richer logging hooks, and benchmark notebooks.

## Releasing
1. Update `CHANGELOG.md` under `[Unreleased]` with the changes being shipped.
2. Bump the version in `pyproject.toml` following Semantic Versioning.
3. Run the lint + test suite (`ruff`, `black`, `pytest`) and ensure CI is green.
4. Tag the release (`git tag vX.Y.Z && git push --tags`) and attach any artefacts (plots, notebooks) to the GitHub release notes.
