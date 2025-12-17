# Contributing Guide

Thank you for considering a contribution! This project hosts multiple TEKNOFEST prototypes (SONIC, AgroScan). We follow a lightweight, quality-first workflow.

## Getting Started
1) Clone and create a virtualenv: `python -m venv .venv && source .venv/bin/activate`
2) Install dev deps: `pip install -r requirements-dev.txt`
3) Pre-commit (optional): `pre-commit install`

## Running Checks
- Format: `black .`
- Lint: `ruff check .`
- Types: `mypy sonic/src tools`
- Tests: `pytest --cov=sonic --cov=tools`

## Development Workflow
- Create a feature branch from `main`.
- Keep changes small and focused; update/add tests alongside code.
- Ensure CI passes before opening a PR.

## Commit & PR Guidelines
- Use clear commit messages (imperative mood).
- Add context in PR description: what, why, how tested.
- Reference related issues/roadmap items.

## Issue Reporting
- Provide reproduction steps, expected vs actual behavior, logs or screenshots.
- Label appropriately (bug, enhancement, docs, demo, infra).

## Security
- Do not commit secrets or private data. Report vulnerabilities privately via email.
