# Contributing

Thanks for contributing to movie-genre-classifier.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Development Workflow

1. Create a feature branch from `main`.
2. Keep changes scoped to one purpose per commit.
3. Run tests before opening a PR.
4. Update `README.md` or docs when behavior changes.

## Testing

```bash
pytest -q
```

For focused testing:

```bash
pytest -q tests/test_models.py tests/test_predict.py
```

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] New or changed behavior is documented.
- [ ] No large generated artifacts are added.
- [ ] Commit messages are clear and concise.
