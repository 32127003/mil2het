# T-005 - scbiomarker: arbitrary multi-view gene embedding input

## Status
- [x] Planned
- [x] In Progress
- [ ] Done

## Objective
Expose the existing FIND multi-view prior path as a user-facing feature that accepts arbitrary named gene embedding views. Users should be able to provide one or more precomputed gene embeddings without being locked to the current `GPT/node2vec/ESM3` naming pattern.

## Dependencies
- [ ] Depends on T-002 and T-003.
- [ ] Should start after the public input contract and config representation for embedding views are defined.
- [ ] Can run in parallel with T-004 once T-003 is complete.
- [ ] Unblocks T-006 and T-007.

## Scope
- [ ] Generalize the current `prior_view_sources` and embedding-path handling in training and biomarker helpers.
- [ ] Define one explicit, documented input contract for named embedding views and gene alignment.
- [ ] Reuse existing `load_prior_embeddings_by_view` and related helper logic where possible.
- [ ] Preserve fail-fast behavior for missing essential inputs and mismatched gene coverage.

## Out of Scope
- [ ] Generating embeddings from raw sequences or external models.
- [ ] Supporting every possible file format on day one.

## Proposed Plan
1. [ ] Identify the narrowest stable on-disk/API representation for embedding views.
2. [ ] Normalize arbitrary view names into the existing prior interface and merged embedding builders.
3. [ ] Add tests for single-view and multi-view user-supplied embeddings with deterministic gene alignment.

## Acceptance Criteria
1. [ ] Users can pass 1-N named embedding views without editing code.
2. [ ] Training and biomarker phases consume the same normalized view definition.
3. [ ] Missing genes, empty views, or incompatible dimensions fail loudly with clear guidance.

## Validation
- [ ] Add tests that feed synthetic embeddings through `load_prior_embeddings_by_view` and the train/biomarker helper paths.
- [ ] Verify the resulting view tensors have the expected shape and stable gene ordering.

## Progress Log
- 2026-03-18: Task skeleton created after confirming the current code already supports multi-view priors internally but still assumes hardcoded source names and config wiring.
- 2026-03-18: First atomic unit completed on the train/biomarker seam. `scripts/train.py` and `modules/biomarker.py` now resolve view names from generic mapping inputs before legacy lists, and missing genes in embedding views now fail fast instead of being zero-filled. Smoke coverage was updated to use arbitrary view names and strict missing-gene failures.
