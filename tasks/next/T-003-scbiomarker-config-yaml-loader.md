# T-003 - scbiomarker: generic YAML config and override loader

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Replace the current dataset-switching config pattern with a generic config layer that can drive the full pipeline for arbitrary user data. The new config path must match the precedence described in `TODO.md`: defaults, then `--config`, then explicit CLI overrides.

## Dependencies
- [ ] Depends on T-002.
- [ ] Can start only after the workflow contract, config precedence, and public input names are frozen.
- [ ] Unblocks T-004, T-005, T-006, T-007, and T-008.

## Scope
- [x] Add a user-facing config schema under `scbiomarker/` rather than relying on dataset-specific dictionaries in `configs/config.py`.
- [x] Support a default config file plus an external YAML file passed by path.
- [x] Implement override precedence for common runtime fields such as epochs, learning rate, output directory, and phase flags.
- [x] Preserve the existing legacy fields needed by downstream training and biomarker code during the migration.
- [x] Keep the solution minimal and explicit rather than building a large framework.

## Out of Scope
- [ ] Building the end-to-end CLI itself.
- [ ] Removing all legacy config code in one step.

## Proposed Plan
1. [x] Design the smallest config schema that covers the current pipeline and the TODO requirements.
2. [x] Implement config loading, normalization, and override merging in `scbiomarker/`.
3. [x] Add compatibility shims so the existing training and analysis internals can consume the normalized config.

## Acceptance Criteria
1. [x] A default YAML-backed config exists and is loadable from package code.
2. [x] Users can pass `--config <path>` and then override selected fields with direct flags.
3. [x] The normalized config can represent explicit user-provided data paths and column names without requiring `--dataset`.

## Validation
- [x] Add tests for config loading and precedence.
- [x] Exercise a smoke path equivalent to `mil2het --config custom.yml --epochs 20 --lr 0.02 --skip_analysis`.

## Implementation Notes
- Added `scbiomarker/default_config.yaml` as the user-facing default config resource.
- Added `scbiomarker/config.py` for:
  - default YAML loading
  - external YAML loading
  - deep merge with explicit override precedence
  - normalized public-to-legacy config adaptation
  - reusable CLI-style argument parsing for future `T-008`
- Preserved downstream compatibility by deriving legacy runtime fields such as:
  - `protein_embedding_paths`
  - `protein_embedding_sources`
  - `prior_view_sources`
  - `splits_directory`
  - `experiment_root`
  - `biomarker_run_dir`
  - `biomarker_output_dir`
- Added YAML support to existing helper loaders in `modules/utils.py` and `modules/biomarker.py` so legacy runtime consumers can accept the new config format during the migration.
- Added smoke coverage in `tests/smoke_test_config_endpoints.py` and extended the endpoint smoke suite to include it.

## Progress Log
- 2026-03-18: Task skeleton created from the gap between `TODO.md` and the current dataset-specific config dictionaries.
- 2026-03-18: Added a package-default YAML config, normalized loader/merge helpers, CLI-style override parsing, YAML compatibility shims for biomarker/util loaders, and smoke coverage.
