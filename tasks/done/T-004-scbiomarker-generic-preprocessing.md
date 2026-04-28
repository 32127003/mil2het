# T-004 - scbiomarker: generic split and preselection inputs

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Decouple split generation and preselection from hardcoded dataset names so they can run on any user-supplied `.h5ad` file. The output should preserve the current split logic and split-train-only DEG/NP behavior while accepting explicit file and column inputs.

## Dependencies
- [ ] Depends on T-002 and T-003.
- [ ] Should start after the workflow input contract and normalized config layer exist.
- [ ] Can run in parallel with T-005 once T-003 is complete.
- [ ] Unblocks T-007.

## Scope
- [x] Refactor `scripts/split_dataset.py` to accept explicit paths and columns instead of only `--dataset`.
- [x] Refactor `scripts/preselection.py` the same way, including explicit output directories.
- [x] Preserve patient-fold logic, optional sample-level handling, and existing output artifacts.
- [x] Keep the implementation centered in `scbiomarker/` as much as possible, with script wrappers calling into package functions.

## Out of Scope
- [ ] Training and biomarker orchestration.
- [ ] Generating or validating external gene embeddings.

## Proposed Plan
1. [x] Extract reusable preprocessing functions from the current script `__main__` blocks.
2. [x] Thread explicit `h5ad`, column, and output-path inputs through the split/preselection flow.
3. [x] Add a small synthetic-data smoke test that proves the generic path works without dataset switches.

## Acceptance Criteria
1. [x] Split creation no longer requires `if args.dataset == ...` config branching.
2. [x] Preselection can run from explicit inputs and still emit split-aware DEG/NP outputs.
3. [x] The preprocessing outputs remain usable by later training and biomarker phases.

## Validation
- [x] Add smoke tests that create a tiny AnnData object, write it to `.h5ad`, and run the generic split/preselection path.
- [x] Verify output files match the expected split and preselection directory layout.

## Implementation Notes
- `scripts/split_dataset.py` now supports both:
  - legacy dataset preset mode via `--dataset`
  - explicit mode via `--adata-path`, explicit columns, and explicit output directory
- Sample-level split grouping now keys off `sample_column` directly instead of a hard `dataset == "asthma_ext"` gate. The special 4/2/2 `asthma_ext` compatibility rule is still preserved for that legacy dataset name.
- `scripts/preselection.py` now has a generic `run_split_preselection(config)` runner that accepts explicit `.h5ad`, PPI, split directory, and output directory inputs.
- The `preselection(...)` function now accepts explicit DEG/RWR parameters so callers no longer need to inject a module-global `config`.
- Package wrappers export the new preprocessing helpers so later tasks can call them through `scbiomarker.split_dataset` and `scbiomarker.preselection`.
- Artifact naming/layout remains compatible:
  - split files still use `{dataset_name}_idx_{fold}.pkl`
  - preselection outputs still use `split_<n>/DEG` and `split_<n>/NP`

## Progress Log
- 2026-03-18: Task skeleton created after confirming both preprocessing scripts still derive paths from `config.dataset`.
- 2026-03-18: Added explicit-input split and preselection runners, removed the hidden preselection global-config dependency, and extended smoke coverage for generic `.h5ad` flows.
