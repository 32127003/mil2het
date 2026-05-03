# T-006 - scbiomarker: reusable training and analysis phase APIs

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Extract reusable package-level callables for the two main phases of the method: training and biomarker analysis. This task should move the project away from script-only `__main__` flows while preserving the current checkpoints, logs, and biomarker outputs.

## Dependencies
- [ ] Depends on T-002, T-003, and T-005.
- [ ] Should start after config normalization exists and the embedding-view input contract is settled.
- [ ] Does not need to wait for T-004, because it can still target existing preprocessing artifacts during extraction.
- [ ] Unblocks T-007.

## Scope
- [x] Extract callable train-phase entrypoints from `scripts/train.py`.
- [x] Extract callable analysis-phase entrypoints from `modules/biomarker.py` and any required helpers.
- [x] Remove mandatory dataset-switch branching from the runtime path.
- [x] Support phase-only execution and checkpoint-driven analysis in the package API.

## Out of Scope
- [ ] The final consolidated end-to-end API.
- [ ] The public CLI parser and console script wiring.

## Proposed Plan
1. [x] Isolate pure runtime functions from parser/bootstrap code.
2. [x] Define explicit input and return contracts for training and analysis phases.
3. [x] Keep thin backward-compatible script wrappers that delegate to the package layer.

## Acceptance Criteria
1. [x] Training can be invoked from Python without entering the current script `__main__` path.
2. [x] Biomarker analysis can be invoked from Python with either a freshly produced or pre-existing checkpoint/run directory.
3. [x] Existing run artifacts remain intact: checkpoints, logs, patient predictions, and biomarker outputs.

## Validation
- [x] Extend smoke tests to call the new package-level train/analysis functions directly.
- [x] Verify artifact directories still contain the expected checkpoint and output files.

## Implementation Notes
- `scripts/train.py`
  - now exposes `run_training_phase(config, *, device=None)` and keeps the CLI path as a thin wrapper through `build_train_arg_parser`, `build_train_config_from_cli_args`, `load_train_config_from_cli`, and `main`
  - returns the training artifact mapping so later orchestration can reuse `output_dir`, checkpoints, logs, and metrics without re-entering `__main__`
- `modules/biomarker.py`
  - now exposes `run_analysis_phase(run_dir, *, config_path="", output_dir="", pathway_path="", gpu_index=None) -> dict`
  - keeps `parse_args()` and `main()` as a thin wrapper around the new runtime callable
- `scbiomarker/train.py` and `scbiomarker/biomarker.py`
  - export the new phase APIs so package users can call them directly
- Smoke coverage now exercises both phase APIs directly:
  - `tests/smoke_test_train_endpoints.py`
  - `tests/smoke_test_biomarker_endpoints.py`

## Progress Log
- 2026-03-18: Task skeleton created after confirming the current train and biomarker flows are still centered on script/module main blocks.
- 2026-03-18: Extracted `run_training_phase` from `scripts/train.py`, exported it through `scbiomarker.train`, and added a lightweight direct-call smoke test for the train-phase seam.
- 2026-03-18: Extracted `run_analysis_phase` from `modules/biomarker.py`, exported it through `scbiomarker.biomarker`, and added a lightweight direct-call smoke test for the analysis-phase seam.
