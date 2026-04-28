# T-007 - scbiomarker: end-to-end Python pipeline API

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Provide one user-facing Python API that runs the intended workflow end-to-end or phase-by-phase from code. This is the package surface that should satisfy the imported-module usage described in `TODO.md`.

## Dependencies
- [ ] Depends on T-004, T-005, and T-006.
- [ ] Also assumes T-002 and T-003 are already complete through those prerequisite tasks.
- [ ] Should start only after generic preprocessing and reusable phase APIs both exist.
- [ ] Unblocks T-008 and T-009.

## Scope
- [x] Add a top-level pipeline function under `scbiomarker/` that orchestrates preprocessing, training, and analysis.
- [x] Accept the explicit workflow inputs settled in T-002, including `.h5ad`/AnnData, columns, PPI, optional condition/label, and named gene embedding views.
- [x] Return a structured result containing key artifact paths and runtime metadata.
- [x] Keep orchestration logic separate from low-level training/model code.

## Out of Scope
- [ ] CLI argument parsing and shell UX details.
- [ ] Rewriting core model internals beyond what is needed to expose the workflow.

## Proposed Plan
1. [x] Design the pipeline function signature around the normalized config and phase APIs.
2. [x] Implement orchestration for full runs plus partial runs (`train_only`, `analysis_only`).
3. [x] Add examples and tests that exercise the API from Python directly.

## Acceptance Criteria
1. [x] A Python caller can run the full pipeline from package code without calling repository scripts.
2. [x] A Python caller can skip analysis or start at analysis from a pre-trained checkpoint/run directory.
3. [x] The API surface is small, explicit, and documented enough to serve as the primary imported-module interface.

## Validation
- [x] Add a Python-level smoke test using a tiny synthetic or fixture-based dataset.
- [x] Verify the returned result object exposes the artifact paths needed by downstream users.

## Progress Log
- 2026-03-18: Task skeleton created from the absence of any current end-to-end package API in `scbiomarker/`.
- 2026-03-18: Added `scbiomarker.pipeline.run_pipeline` and `PipelineResult`, exported them from the package root, and added smoke coverage for full, train-only, and analysis-only flows.
- 2026-03-18: The pipeline now writes an effective `workflow_config.yaml` snapshot for analysis calls so the public API does not depend on the legacy `*_config.py` discovery path.
