# T-009 - scbiomarker: end-to-end validation and test-project docs

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Prove that the pip-installable module is usable from the separate `test_biomarker` project and document the exact workflow. This task closes the loop on packaging, manual dependency notes, Python API usage, and CLI usage.

## Dependencies
- [x] Depends on T-007 and T-008.
- [x] Should start only after the final Python API and CLI surfaces are both in place.
- [x] This is the closing validation/documentation task for the current backlog slice.

## Scope
- [x] Extend smoke coverage beyond endpoint existence to the new config/API/CLI workflow.
- [x] Update the test-project documentation with refresh-install, dependency, API, and CLI steps.
- [x] Verify the `py_module` branch install flow remains the source of truth.
- [x] Keep dependency documentation fail-fast and explicit, consistent with the existing import-error guidance.

## Out of Scope
- [ ] Large-scale benchmark runs or parameter sweeps.
- [ ] Hiding missing GPU stack dependencies behind silent fallbacks.

## Proposed Plan
1. [x] Add installable smoke tests for the new user-facing paths.
2. [x] Update docs in the test project to match the final workflow.
3. [x] Run the refresh-install and smoke-test loop from the `test` environment.

## Acceptance Criteria
1. [x] The test project documents both Python API and CLI usage for the packaged module.
2. [x] A fresh install from `git+ssh://git@github.com/32127003/scbiomarker.git@py_module` exposes and exercises the final interface.
3. [x] Validation covers package import, generic config loading, and at least one end-to-end smoke path.

## Validation
- [x] `conda run -n test pip uninstall -y scbiomarker`
- [x] `conda run -n test pip install git+ssh://git@github.com/32127003/scbiomarker.git@py_module`
- [x] `conda run -n test python tests/run_all_endpoint_smoke_tests.py`
- [x] Add and run dedicated smoke commands for the new end-to-end API/CLI path.

## Progress Log
- 2026-03-18: Task skeleton created after reviewing `tests/README.md` and the current endpoint-only smoke coverage.
- 2026-03-20: Added a dedicated step-1 TODO example smoke script under `tests/` that exercises both the public CLI train-only path and the public Python-import train-only path, and wired it into the canonical smoke runner.
- 2026-03-20: Synced the canonical `tests/` assets into the parent-level `test_biomarker` smoke project, including the new step-1 example script and refreshed README instructions.
- 2026-03-20: Verified fresh install in the `test` conda env via `pip uninstall` + `pip install git+ssh://git@github.com/32127003/scbiomarker.git@py_module`, ran the installed-project smoke suite successfully, and confirmed the installed `scbiomarker` console entrypoint exists and responds to `--help`.
