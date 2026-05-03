# T-011 - mil2het: Read the Docs hosted documentation

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Enable Read the Docs to build and serve the existing Sphinx documentation from this repository with a reproducible hosted build configuration.

## Scope
- [x] Add a repository-root `.readthedocs.yaml` using Read the Docs config version 2, `build.os`, `build.tools.python`, and `sphinx.configuration: docs/conf.py`.
- [x] Decide and implement the install strategy for the hosted build: either native Read the Docs `uv` install using `pyproject.toml` dependency groups, or a dedicated docs requirements/extra path if that is more compatible with the project and RTD.
- [x] Ensure the docs build installs only documentation/runtime import dependencies needed for autodoc, not the CUDA-heavy training stack from `requirements.txt`.
- [x] Keep the existing Sphinx source under `docs/` and keep generated `docs/_build/` artifacts ignored.
- [x] Update a maintainer-facing doc page or README section with the Read the Docs project setup steps, expected build command, and troubleshooting notes for optional Torch imports.
- [x] Required validation level is a local clean Sphinx build plus a local command that mirrors the selected Read the Docs install/build path as closely as possible.

## Out of Scope
- [ ] Do not publish credentials, configure a custom domain, or make account-level Read the Docs changes from the repo.
- [ ] Do not add GPU, CUDA, PyTorch, or `torch-scatter` installation to the Read the Docs build unless autodoc coverage is intentionally expanded to require those objects without mocks.
- [ ] Do not redesign the Sphinx page structure or rewrite the API documentation created in T-010 beyond hosted-build compatibility fixes.
- [ ] Do not commit generated HTML, PDF, ePub, or `htmlzip` artifacts.

## Proposed Plan
1. [x] Confirm the final hosted-build strategy against current Read the Docs support: `.readthedocs.yaml` at repo root, Ubuntu image, Python version, Sphinx config path, and either `python.install` with `method: uv`/`command: sync`/`groups: [dev]` or a docs-specific requirements/extra install.
2. [x] If using the `uv` strategy, verify `uv.lock` is committed and that the `dev` dependency group is acceptable for hosted docs; if not, create a narrower docs dependency group or `docs/requirements.txt` containing `sphinx`, `sphinx-autodoc-typehints`, `myst-parser`, and `shibuya`.
3. [x] Add `.readthedocs.yaml` with `version: 2`, `build.os: ubuntu-24.04`, `build.tools.python: "3.12"`, `sphinx.configuration: docs/conf.py`, and warning behavior aligned with the current strict local build.
4. [x] Audit `docs/conf.py` for Read the Docs behavior, including local package import paths, `autodoc_mock_imports` for optional Torch modules, and no reliance on local absolute paths or generated build artifacts.
5. [x] Add concise maintainer instructions describing how to import the repository into Read the Docs, which branch should build, where the config file lives, and how to reproduce failures locally.
6. [x] Run the selected local validation commands from a clean environment path and fix any missing dependency, import, or warning failures.

## Acceptance Criteria
1. [x] A valid `.readthedocs.yaml` exists at the repository root and points Read the Docs at `docs/conf.py`.
2. [x] Read the Docs can install the docs build dependencies without pulling the CUDA-oriented `requirements.txt` stack.
3. [x] The hosted-build configuration is reproducible from committed files, including any lockfile, docs requirements file, or project dependency group it depends on.
4. [x] Local validation builds the docs with the same Sphinx config and warning policy intended for Read the Docs.
5. [x] Maintainer documentation explains the Read the Docs import/setup steps and how to diagnose missing dependency or optional Torch autodoc issues.

## Validation
- [x] Run `env -u VIRTUAL_ENV uv run sphinx-build -E -b html -W --keep-going docs docs/_build/html` from `/home/cobalt/Projects/bio/mil2het_test/mil2het`.
- [x] Run the local equivalent of the selected RTD install path, for example `uv sync --group dev` for a `method: uv`/`command: sync` config, or `uv pip install -r docs/requirements.txt -e .` for a requirements-based config.
- [x] Validate the Read the Docs YAML shape with a parse check such as `uv run python -c "import yaml; yaml.safe_load(open('.readthedocs.yaml'))"` and confirm required keys are present.
- [x] Confirm `git status --short --ignored docs docs/_build .readthedocs.yaml` shows `docs/_build/` ignored and no generated hosted output tracked.

## Progress Log
- 2026-04-28: Task created after reviewing the existing Sphinx docs, project dependency layout, and current Read the Docs configuration guidance for Sphinx and native `uv` installs.
- 2026-04-28: Completed Read the Docs config using native `uv sync --group docs`, added hosted-doc maintainer notes, validated YAML shape, and passed strict clean Sphinx build.
