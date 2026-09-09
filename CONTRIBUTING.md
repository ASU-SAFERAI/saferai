# Contributing to SAFER AI

Thank you for your interest in contributing to SAFER AI.

SAFER AI is an open-source project supporting responsible, safe, and effective
uses of AI in higher education. Contributions from developers, researchers,
educators, students, accessibility specialists, and institutional partners
are welcome.

We expect all participants to be respectful, inclusive, and professional in
project spaces. Harassment and discriminatory behavior are not tolerated.

## Ways to Contribute

You can contribute by:

- Fixing bugs or improving performance
- Proposing and implementing features
- Improving documentation and examples
- Adding or improving tests
- Contributing evaluation metrics, benchmarks, or test cases
- Improving accessibility and usability
- Reporting safety, privacy, fairness, or security concerns
- Sharing higher-education use cases

Look for issues labeled `good first issue` or `help wanted` if you are new
to the project.

SAFER AI is a monorepo of three independently versioned and installable
components (`pre-deploy/`, `in-production/`, `post-deploy/`). Most changes
touch only one of them. See the [README](README.md) for the architecture
overview and each component's README for details.

## Before You Begin

For small bug fixes or documentation improvements, you may submit a pull
request directly.

Before beginning a substantial feature, architectural change, new dataset,
or evaluation methodology:

1. Search existing issues and pull requests.
2. Open an issue describing the proposed change.
3. Explain the problem, intended users, and expected impact.
4. Wait for maintainer feedback before investing significant effort.

This helps avoid duplicate work and ensures that the proposal aligns with
SAFER AI's scope.

## Development Setup

Install the component you are working on by following its README (linked from
the [main README](README.md)). Because the components pin different versions
and dependencies, use a separate virtual environment for each one.

A few notes specific to contributors:

- **`pre-deploy/`** uses [pre-commit](https://pre-commit.com/) hooks. Install
  them once after cloning so formatting, safety, and commit-message checks run
  automatically:

  ```bash
  cd pre-deploy
  pip install pre-commit
  pre-commit install
  ```

- **`post-deploy/`** ships dev tools (`pytest`, `ruff`) in its extras. Install
  the `dev` or `all` extra so they are available:

  ```bash
  cd post-deploy
  uv pip install -e ".[all]"   # or: pip install -e ".[dev]"
  ```

## Testing

Run the test suite for the component you changed and make sure it passes
before opening a pull request.

```bash
# pre-deploy
cd pre-deploy && pytest

# post-deploy
cd post-deploy && pytest

# in-production (local handler tests)
cd in-production && python test_main.py --test apigw-event
```

Add or update tests for any behavior you change. New metrics, benchmarks, and
bug fixes should come with tests that demonstrate the change.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) and keep changes consistent
  with the surrounding code.
- `post-deploy` is linted with [ruff](https://docs.astral.sh/ruff/)
  (line length 120). Run it before committing:

  ```bash
  cd post-deploy && ruff check . && ruff format .
  ```

- For `pre-deploy`, the pre-commit hooks handle end-of-file fixing, trailing
  whitespace, merge-conflict checks, and secret detection. Do not commit
  directly to `main`.

## Commit Messages

The project uses [Conventional Commits](https://www.conventionalcommits.org/).
Structure the subject line as `type(scope): description`, for example:

```
fix(post-deploy): handle empty input in keyword_search
feat(pre-deploy): add MARBLE bias evaluation step
docs: clarify in-production alerting configuration
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`. In
`pre-deploy` this is enforced by a commit-msg hook.

## Pull Requests

1. Fork the repository and create a branch from `main`
   (e.g. `fix/keyword-search-empty-input`).
2. Make your change, add tests, and run the relevant test suite and linters.
3. Keep the pull request focused on a single concern; note which component(s)
   it touches.
4. Write a clear description covering what changed, why, and how you verified
   it.
5. Link any related issue.

A maintainer will review your pull request and may request changes. Please be
responsive to feedback so we can merge your contribution smoothly.

## Data and Privacy

SAFER AI is designed to work with synthetic and de-identified data. Do not
commit identifiable student records, real user interactions, credentials, or
other sensitive data. Use synthetic examples in tests and documentation.

## Reporting Security Issues

Do not report security vulnerabilities through public issues or pull requests.
Follow the private process described in [SECURITY.md](SECURITY.md).

For non-security safety, privacy, or fairness concerns, open a normal issue
with enough detail to reproduce or understand the problem.
