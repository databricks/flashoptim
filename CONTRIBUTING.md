# Contributing to FlashOptim

Thanks for your interest in contributing to FlashOptim! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/databricks/flashoptim.git
cd flashoptim
pip install -e ".[dev]"
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, enforced via [pre-commit](https://pre-commit.com/) hooks.

Install the pre-commit hooks:

```bash
pre-commit install
```

You can run all checks manually at any time:

```bash
pre-commit run --all-files
```

## Running Tests

```bash
python -m pytest test/
```

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes and ensure `pre-commit run --all-files` passes.
3. Add or update tests as appropriate.
4. Open a pull request with a clear description of the change.
