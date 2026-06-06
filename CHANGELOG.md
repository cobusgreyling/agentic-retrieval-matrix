# Changelog

All notable changes to Agentic Retrieval Matrix (ARM) will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`): lint + format + pytest across Python 3.11/3.12, vector extra smoke test, package build + wheel install check, matrix smoke run.
- Comprehensive new tests: `test_grader.py`, `test_cli.py` (Typer CliRunner), enhanced matrix fixture coverage, placeholder `test_vector_retriever.py`.
- `arm --version` / `-V` flag (and `arm version` not needed).
- `arm single` and `arm longmem` now save results to disk and report the path consistently with `matrix`.
- `py.typed` marker (PEP 561) for type checkers.
- URLs in pyproject (Homepage, Repository, Paper, Bug Tracker).
- This `CHANGELOG.md`.

### Changed
- **Major hygiene & quality pass**:
  - All code formatted with `ruff format`.
  - All auto-fixable ruff lint issues resolved (UP045 Optional→|None, datetime.UTC, F401, E501, etc.).
  - Switched `RetrieverKind` / `DeliveryKind` / `HarnessKind` to `enum.StrEnum` (cleaner, Python 3.11+).
  - Moved `numpy` from core dependencies into the `[vector]` extra (only required by the vector path).
- Improved inline delivery hit formatting (slightly more compact, still under line-length).
- Updated `ReactHarness` docstring to clearly document that it is a *delivery-aware extractive baseline* (oracle) rather than a true tool-calling agent. This explains why inline vs file gaps are currently small.
- README refreshed: clearer example output, new "Baseline & Limitations" section, concrete extending code snippets, development commands, quick-start version check.
- Version bumped to 0.2.0 to reflect the quality, test, CI, and documentation improvements.
- `pyproject.toml` metadata expanded; sdist includes examples/scripts.

### Fixed
- CLI now imports `__version__` directly from the package.
- Minor: cleaned a long line in delivery presentation that violated the 100-char limit.
- Test matrix now asserts on details shape and performs save roundtrip.

### Notes
- The fixture now reports 100% for both inline and file under the current baseline harness (the harness was already strong; previous README example of 66.7% reflected an earlier extractive implementation).
- LongMemEval converter remains a stub (as designed). Real converter + 100+ question runs are the primary research extension point.
- No breaking API changes for retriever/delivery/harness authors.

## [0.1.0] - 2026-06 (initial)

- Initial public release of the factorial benchmark (retriever × delivery × harness).
- Grep + Vector retrievers, Inline + File delivery, minimal React baseline harness.
- Memory materialization to files for first-class lexical search.
- Loader for custom benchmarks + LongMemEval normalized layout hook.
- `arm matrix`, `arm single`, `arm longmem` commands + rich table output.
- Grader with literal + token-overlap heuristics (no LLM required).
- Results written as timestamped JSON with per-question traces.
