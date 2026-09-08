# AGENT.md

## Project Overview

FedClsimb is a PyTorch-based federated learning experiment codebase. It trains image classifiers under different client data partition strategies and compares multiple FL algorithms.

## Configuration

Algorithm config skeletons are under `src/config/*.json`.

## Data Pipeline

Data distribution is implemented in `src/train_tools/preprocessing/datasetter.py`.

## Models

Model creation is centralized in `src/train_tools/utils.py`.

## Algorithm Architecture

Shared base classes:

- `src/algorithms/BaseServer.py`
- `src/algorithms/BaseClientTrainer.py`

`BaseServer` owns the global training loop.
`BaseClientTrainer` owns standard local training.

Each algorithm subpackage follows the same rough layout:

- `Server.py`
- `ClientTrainer.py`
- Optional `criterion.py`
- Optional `utils.py`
- `__init__.py`

## Metrics And Logging

Metrics helpers are in `src/algorithms/measures.py`.

## Adding A New Algorithm

To add a new algorithm consistently with this codebase:

1. Create `src/algorithms/<algo>/Server.py`, `ClientTrainer.py`, and `__init__.py`.
2. Subclass `BaseServer` and `BaseClientTrainer` unless the algorithm needs a custom round loop.
3. Add any custom loss to `criterion.py` and helper functions to `utils.py`.
4. Export the package from `src/algorithms/__init__.py`.
5. Register the server in `ALGO` inside `src/main.py`.
6. Add a config skeleton in `src/config/<algo>.json`.
7. Ensure the selected model supports any extra forward arguments required by the algorithm.