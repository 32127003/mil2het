# T-005 - scbiomarker: arbitrary multi-view gene embedding input

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Expose the existing FIND multi-view prior path as a user-facing feature that accepts arbitrary named gene embedding views. Users should be able to provide one or more precomputed gene embeddings without being locked to the current `GPT/node2vec/ESM3` naming pattern.

## Dependencies
- [ ] Depends on T-002 and T-003.
- [ ] Should start after the public input contract and config representation for embedding views are defined.
- [ ] Can run in parallel with T-004 once T-003 is complete.
- [ ] Unblocks T-006 and T-007.

## Scope
- [x] Generalize the current `prior_view_sources` and embedding-path handling in training and biomarker helpers.
- [x] Define one explicit, documented input contract for named embedding views and gene alignment.
- [x] Reuse existing `load_prior_embeddings_by_view` and related helper logic where possible.
- [x] Preserve fail-fast behavior for missing essential inputs and mismatched gene coverage.

## Out of Scope
- [ ] Generating embeddings from raw sequences or external models.
- [ ] Supporting every possible file format on day one.

## Proposed Plan
1. [x] Identify the narrowest stable on-disk/API representation for embedding views.
2. [x] Normalize arbitrary view names into the existing prior interface and merged embedding builders.
3. [x] Add tests for single-view and multi-view user-supplied embeddings with deterministic gene alignment.

## Acceptance Criteria
1. [x] Users can pass 1-N named embedding views without editing code.
2. [x] Training and biomarker phases consume the same normalized view definition.
3. [x] Missing genes, empty views, or incompatible dimensions fail loudly with clear guidance.

## Validation
- [x] Add tests that feed synthetic embeddings through `load_prior_embeddings_by_view` and the train/biomarker helper paths.
- [x] Verify the resulting view tensors have the expected shape and stable gene ordering.

## Implementation Notes
- `modules/utils.py`
  - now resolves arbitrary embedding view names from configured path mappings before falling back to the built-in `GPT`, `node2vec`, and `ESM3` defaults
  - raises a clear error when a non-built-in view name is used without a configured path
- `scripts/train.py`
  - uses the configured view set when filtering NP genes for embedding coverage instead of the old hardcoded built-in list
  - fails fast when a required embedding view is missing genes from the selected gene set
- `modules/biomarker.py`
  - accepts the same normalized configured view definitions during analysis-time model reconstruction
  - fails fast on missing genes instead of silently zero-filling
- Added smoke coverage for:
  - arbitrary named embedding views in both train and biomarker helper paths
  - fail-fast behavior when a configured embedding view is missing required genes

## Progress Log
- 2026-03-18: Task skeleton created after confirming the current code already supports multi-view priors internally but still assumes hardcoded source names and config wiring.
- 2026-03-18: First atomic unit completed on the train/biomarker seam. `scripts/train.py` and `modules/biomarker.py` now resolve view names from generic mapping inputs before legacy lists, and missing genes in embedding views now fail fast instead of being zero-filled. Smoke coverage was updated to use arbitrary view names and strict missing-gene failures.
