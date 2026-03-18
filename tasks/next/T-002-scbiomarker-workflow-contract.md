# T-002 - scbiomarker: workflow contract and paper parity

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Turn `TODO.md` plus the current repository behavior into an explicit workflow contract before refactoring. This task should settle the module/API/CLI surface, the two-phase execution model, and which paper-derived behaviors must remain unchanged.

## Dependencies
- [ ] None. This is the contract-setting task for the rest of the backlog.
- [ ] Unblocks T-003 through T-008.

## Scope
- [x] Audit the current script-style entrypoints in `scripts/split_dataset.py`, `scripts/preselection.py`, `scripts/train.py`, and `modules/biomarker.py`.
- [x] Define the minimum user inputs for the generic workflow: `.h5ad`, patient column, cell type column, optional condition/label column, PPI network, and named gene embedding views.
- [x] Define phase boundaries and artifacts for `split -> preselection -> training -> analysis`.
- [x] Define the public interface contract for Python API usage and CLI usage, including config precedence.
- [x] Record paper checks that still need direct PDF verification once PDF tooling is fixed.

## Out of Scope
- [ ] Refactoring training or analysis logic.
- [ ] Implementing the new CLI or API.

## Proposed Plan
1. [x] Compare `TODO.md` against the actual assumptions embedded in the current codebase.
2. [x] Write the canonical contract for inputs, outputs, phase control, and naming.
3. [x] Freeze the migration order so later tasks do not re-open interface questions.

## Acceptance Criteria
1. [x] The task resolves the command/package naming question and the phase-control semantics (`train_only`, `analysis_only`, checkpoint-driven analysis).
2. [x] The task documents which current paths/config fields are legacy internals versus part of the new public contract.
3. [x] The task records any paper-specific items that remain unverified because `pypdf` is currently missing in both the base shell and the `scbiomarker` conda env.

## Validation
- [x] Cross-check the written contract against `TODO.md`, `BACKLOG.md`, and the current parser/config code paths.
- [x] Confirm the contract can cover the existing asthma-style workflow without losing current artifacts.

## Resolved Contract

### Naming
- Python package name is frozen as `scbiomarker`.
- The top-level end-to-end public API will be exported from `scbiomarker` rather than introducing a second import package name.
- The installed CLI command will also use the `scbiomarker` name. The `mil2het` examples in `TODO.md` are treated as placeholder UX sketches, not a naming requirement.

### Public Workflow Inputs
- CLI input for the dataset itself is a `.h5ad` path. Python input may be either a `.h5ad` path or an in-memory `AnnData`.
- Required columns for full preprocessing/training runs:
  - `patient_column`
  - `celltype_column`
  - `label_column`
- Optional columns:
  - `sample_column`
    - This replaces the current `dataset == "asthma_ext"` sentinel for patient-sample grouped folding.
  - `treatment_column`
    - This remains optional model metadata, not a prerequisite for the workflow.
- Required external files for full runs:
  - `ppi_path`
  - one or more named gene embedding views
- Gene embedding views are part of the public contract as a mapping of `view_name -> file path`.
  - CLI representation will use repeated explicit flags rather than hardcoded source names.
  - Internal legacy names such as `GPT`, `node2vec`, and `ESM3` remain valid as ordinary user-supplied view names, not reserved keywords.

### Phase Model
- Canonical workflow phases are:
  1. `split`
  2. `preselection`
  3. `training`
  4. `analysis`
- Default behavior for the top-level API/CLI is the full ordered workflow above.
- `train_only` means:
  - run `split -> preselection -> training`
  - do not start biomarker analysis
- `analysis_only` means:
  - skip split, preselection, and training
  - require a previously completed training run directory as the canonical input
- Checkpoint-driven analysis is frozen to `run_dir` semantics for the first public contract.
  - A bare checkpoint path alone is not sufficient public input because analysis also needs the saved config snapshot, preprocessing artifact locations, and model reconstruction metadata.

### Public Config Contract
- Config precedence is frozen as:
  1. package default YAML
  2. external YAML passed via `--config`
  3. explicit CLI flags
- The normalized config must accept explicit user inputs and must not require `--dataset`.
- Public config fields should describe the workflow directly:
  - input `.h5ad`
  - column names
  - `ppi_path`
  - embedding views
  - output root
  - runtime knobs such as `seed`, `num_folds`, `epochs`, `lr`, and phase flags

### Public Outputs and Artifacts
- The public workflow returns or prints a structured view of artifact locations instead of exposing raw internal path math.
- Required artifact groups that must remain intact through the migration:
  - split files
  - split-aware preselection outputs (`DEG`, `NP`, `DEG_zscore_global.tsv`)
  - training run directory with checkpoint, logs, and saved config snapshot
  - analysis output directory with biomarker tables
- The current artifact layout remains a compatibility target during migration so `T-004` and `T-006` can proceed without breaking downstream consumers.

### Legacy Internal Fields vs Frozen Public Surface
- Legacy/internal, not part of the new public contract:
  - `dataset`
  - dataset-switch dictionaries in `configs/config.py`
  - `adata_directory`
  - `splits_directory`
  - `experiment_root`
  - `protein_embedding_sources`
  - `prior_view_sources`
  - `protein_embedding_paths`
  - `split_preselection_subdir`
  - path heuristics that infer directories from dataset names
- Publicly supported, either as direct fields or normalized aliases:
  - `.h5ad` input
  - patient/celltype/label/sample/treatment column names
  - `ppi_path`
  - named embedding views
  - output root / run directory
  - `num_folds`
  - `seed`
  - `epochs`
  - `lr`
  - `train_only`
  - `analysis_only`

### Existing Behavior That Must Be Preserved
- Split generation remains patient-disjoint and optionally patient-sample grouped when `sample_column` is provided.
- Preselection remains split-train-only and continues to emit the current `DEG` and `NP` artifacts used by training and biomarker analysis.
- Training remains the first model phase and continues to emit run snapshots, checkpoints, logs, and patient-level predictions.
- Biomarker analysis remains the second model phase and continues to reconstruct the trained model from the saved run directory.
- Multi-view priors remain part of the core method rather than an optional post-hoc add-on.

### Migration Order Freeze
- `T-003` must establish the normalized config and override precedence before any generic input refactors.
- `T-004` may run in parallel with `T-005` after `T-003`, but it must preserve the current split/preselection artifact contract.
- `T-005` must settle the named embedding-view contract before `T-006`.
- `T-006` must expose reusable train/analysis callables before `T-007` can add a top-level orchestration API.
- `T-008` wraps the stable `T-007` API rather than building a second orchestration path.
- `T-009` validates the final Python API and CLI after the interface stops moving.

### Paper Parity Notes Still Pending Direct PDF Verification
- `pypdf` is still missing in both the base shell and the `scbiomarker` conda environment as of 2026-03-18.
- The following are frozen from current code/TODO review but still need direct paper verification later:
  - exact terminology used for the two-phase workflow
  - whether any paper text imposes stricter semantics on the multiview prior inputs
  - whether any paper-described analysis artifact naming should be surfaced verbatim in docs
- These pending PDF checks do not block the migration contract because the current code and `TODO.md` already agree on the two-phase structure and multiview-prior requirement.

## Progress Log
- 2026-03-18: Task skeleton created from TODO/codebase review; direct PDF extraction is currently blocked by missing `pypdf`.
- 2026-03-18: Workflow contract frozen from `TODO.md` plus current code paths in `scripts/split_dataset.py`, `scripts/preselection.py`, `scripts/train.py`, `modules/biomarker.py`, and `configs/config.py`.
