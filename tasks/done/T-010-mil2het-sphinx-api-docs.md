# T-010 - mil2het: Sphinx API reference documentation

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Create a complete Sphinx documentation page set for the installable `mil2het` package, with a properly organized API reference that reflects the wrapper-facing public surface and separates stable user APIs from lower-level implementation helpers.

## Scope
- [x] Update the existing `mil2het/docs` Sphinx scaffold, including `conf.py`, `index.rst`, and new API reference pages under `mil2het/docs/api/`.
- [x] Configure Sphinx extensions already available in `mil2het/pyproject.toml`, especially autodoc, autosummary, napoleon, intersphinx where useful, myst-parser if Markdown pages are added, and sphinx-autodoc-typehints.
- [x] Define API reference pages for the package facade in `mil2het/__init__.py`, workflow orchestration in `mil2het.pipeline`, config loading in `mil2het.config`, CLI entry points in `mil2het.cli`, split/preselection/train/biomarker wrapper modules, and neural model classes exposed through `CellEncoder`, `MultipleInstanceLearning`, and `prior_interface_find`.
- [x] Decide and document API stability tiers: public user APIs, advanced workflow phase APIs, and internal/legacy implementation modules from `scripts.*` and `modules.*`.
- [x] Add concise narrative pages for installation, quickstart, workflow overview, configuration schema, CLI usage, output artifacts, and analysis-only replay so the API reference is discoverable in context.
- [x] Keep documentation source in the package repo and avoid committing generated `_build` HTML artifacts.
- [x] Required validation level is a clean Sphinx HTML build plus a link/warning pass that treats unresolved autodoc imports, missing references, and duplicate object warnings as actionable failures.

## Out of Scope
- [ ] Do not redesign the `mil2het` public API or rename package modules as part of this docs task.
- [ ] Do not rewrite large docstrings across the legacy implementation unless a small docstring patch is required for Sphinx importability or a critical public API explanation.
- [ ] Do not add new runtime dependencies; use the Sphinx/dev dependencies already declared in `mil2het/pyproject.toml`.
- [ ] Do not publish hosted documentation or configure CI deployment in this task.
- [ ] Do not document every private helper from `scripts.train`, `modules.biomarker`, or `modules.utils`; include implementation helpers only when they are intentionally reachable from wrapper `__all__` surfaces.

## Proposed Plan
1. [x] Audit the existing Sphinx scaffold in `mil2het/docs` and confirm the documentation build command from inside the nested package repo.
2. [x] Update `mil2het/docs/conf.py` with the package import path, project metadata, selected extensions, autosummary generation, autodoc defaults, type-hint rendering, intersphinx mappings, and warning behavior appropriate for a local package build.
3. [x] Replace the placeholder `mil2het/docs/index.rst` with a real landing page and toctrees for user guide pages and API reference pages.
4. [x] Add user guide pages for installation, quickstart, workflow concepts, CLI usage, YAML configuration, generated outputs, and analysis-only replay.
5. [x] Add `mil2het/docs/api/index.rst` that explains API stability tiers and links to focused reference pages.
6. [x] Add an API page for the top-level package facade documenting `run_pipeline`, `PipelineResult`, `GraphCellEncoder`, `TransformerConvCellEncoder`, `GatedAttentionMIL`, `PatientMILAggregator`, and `MultiViewPriorInterfaceFIND`.
7. [x] Add workflow API pages for `mil2het.pipeline`, `mil2het.config`, and `mil2het.cli`, emphasizing `run_pipeline`, config normalization/loading helpers, and CLI parser/entrypoint behavior.
8. [x] Add phase API pages for `mil2het.split_dataset`, `mil2het.preselection`, `mil2het.train`, and `mil2het.biomarker`, documenting the wrapper-level exported functions and noting when they delegate to legacy `scripts.*` or `modules.*` implementations.
9. [x] Add model API pages for `mil2het.CellEncoder`, `mil2het.MultipleInstanceLearning`, and `mil2het.prior_interface_find`, focused on exposed classes rather than every internal tensor helper.
10. [x] Add an internal-reference page or appendix for selected lower-level modules only if Sphinx can render them cleanly and they clarify advanced extension points.
11. [x] Build docs locally, inspect warnings, and either fix docs/import configuration or explicitly exclude unstable/private objects from autodoc.
12. [x] Update README or `TESTING.md` only if needed to point maintainers to the docs build command.

## Acceptance Criteria
1. [x] `mil2het/docs/index.rst` is no longer the Sphinx quickstart placeholder and provides a navigable table of contents for user guide and API reference pages.
2. [x] `mil2het/docs/api/` contains focused API pages for top-level facade, pipeline, config, CLI, workflow phases, and model classes.
3. [x] The API reference clearly distinguishes public package APIs from advanced phase wrappers and internal legacy modules.
4. [x] The reference includes `run_pipeline`, `PipelineResult`, config loaders, CLI entry points, split generation, preselection, training, biomarker analysis, and exposed neural model classes.
5. [x] Sphinx autodoc imports the local package from the nested repo without requiring installation from PyPI or GitHub.
6. [x] The generated documentation does not include `_build` artifacts or large runtime outputs in version control.
7. [x] At least one docs page explains how `--gpu -1`, `train_only`, `analysis_only`, `output_root`, `run_dir`, and `pathway_path` relate to the user-facing workflow.
8. [x] The docs build completes without unresolved critical warnings for included public API pages.

## Validation
- [x] From `/home/cobalt/Projects/bio/mil2het_test/mil2het`, run `uv run sphinx-build -b html docs docs/_build/html`.
- [x] From `/home/cobalt/Projects/bio/mil2het_test/mil2het`, run `uv run sphinx-build -b html -W --keep-going docs docs/_build/html` if third-party import behavior allows warning-as-error mode.
- [x] From `/home/cobalt/Projects/bio/mil2het_test/mil2het`, run `uv run python -c "import mil2het; print(mil2het.__all__)"` and confirm the documented facade names match the package exports.
- [x] Confirm `git status --short` does not show `mil2het/docs/_build/` or other generated HTML artifacts staged for commit.

## Progress Log
- 2026-04-28: Task created with a Sphinx API reference plan based on the existing `mil2het/docs` scaffold, package exports, wrapper modules, and declared documentation dependencies.
- 2026-04-28: Completed the Sphinx page set, API reference, strict warning-as-error docs build, facade export validation, and `_build` ignore check.
