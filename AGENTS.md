# Repository Guidelines

## Project Structure & Module Organization

This fork adds pi0.5 subtask support on top of OpenPI. Core code lives in `src/openpi/`: `models/` and `models_pytorch/` implement models, `training/` contains configs and loaders, `policies/` maps robot observations/actions, `serving/` provides websocket serving, and `shared/` holds utilities. The runtime client is a workspace package in `packages/openpi-client/`. Entry points are in `scripts/`; robot and benchmark examples are in `examples/`; documentation is in `docs/`. Tests are mostly colocated as `*_test.py` under `src/`, `scripts/`, and `packages/`; `run_test/` contains ad hoc/manual checks.

## Build, Test, and Development Commands

- `GIT_LFS_SKIP_SMUDGE=1 uv sync`: install pinned workspace dependencies without downloading large LFS files.
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`: install the repository editable for local development.
- `uv run pytest`: run the configured pytest suite.
- `uv run pytest src/openpi/models/model_test.py`: run a focused test file.
- `uv run ruff check .` and `uv run ruff format .`: lint and format Python code.
- `uv run scripts/compute_norm_stats.py --config-name pi05_libero`: compute normalization statistics.
- `uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite`: launch a training run.
- `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/...`: serve a trained policy.

## Coding Style & Naming Conventions

Use Python 3.11 for the main package and keep `packages/openpi-client` compatible with Python 3.7. Ruff enforces a 120-character line length and single-line import sorting. Use `snake_case` for functions, modules, config names, and CLI flags; use `PascalCase` for classes and dataclasses. Keep config names lowercase, for example `pi05_libero`.

## Testing Guidelines

Use pytest. Name tests `*_test.py` or `test_*.py`, colocated near the code they cover. Mark long-running or hardware-dependent tests with the existing `manual` marker. Prefer focused unit tests for transforms, tokenizers, policies, and loaders before adding expensive integration checks.

## Commit & Pull Request Guidelines

Git history uses short, descriptive commit messages, sometimes with prefixes such as `docs:`. Keep commits scoped and imperative, for example `docs: update subtask README` or `fix policy subtask generation`. Pull requests should include a clear title, concise description, linked issues when applicable, reproduction steps for bug fixes, and test/lint results. Run pre-commit before submitting; it checks `uv-lock`, Ruff linting, and Ruff formatting.

## Security & Configuration Tips

Do not commit model checkpoints, datasets, W&B tokens, private credentials, or local cache paths. Keep large generated artifacts under ignored output directories such as `checkpoints/` or external storage, and document required environment variables in `docs/` or example READMEs.
