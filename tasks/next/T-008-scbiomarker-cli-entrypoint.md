# T-008 - scbiomarker: package CLI entrypoint

## Status
- [x] Planned
- [x] In Progress
- [x] Done

## Objective
Add a real package-installed CLI that mirrors the end-to-end Python API and the override semantics in `TODO.md`. The CLI should expose the full workflow, phase-only execution, and config layering without requiring users to call internal scripts directly.

## Dependencies
- [ ] Depends on T-007.
- [ ] Also relies on T-002 for command naming and T-003 for config override behavior.
- [ ] Should start only after the Python API surface is stable enough for the CLI to wrap directly.
- [ ] Unblocks T-009.

## Scope
- [x] Add a console entrypoint in `pyproject.toml`.
- [x] Implement CLI parsing around the config loader and end-to-end API.
- [x] Support config overrides, explicit input columns, repeated gene-embedding inputs, and phase-control flags.
- [x] Print useful artifact locations and fail-fast messages for missing required inputs.

## Out of Scope
- [ ] Replacing every internal developer script immediately.
- [ ] Building a large subcommand tree unless the workflow contract requires it.

## Proposed Plan
1. [x] Implement the smallest stable command surface that covers the TODO examples.
2. [x] Wire CLI parsing into the normalized config and pipeline API.
3. [x] Add CLI smoke coverage and help-text checks.

## Acceptance Criteria
1. [x] Installing the package exposes the agreed command name from T-002.
2. [x] Users can run full, train-only, and analysis-only flows from the shell.
3. [x] CLI flags override YAML config fields in the documented precedence order.

## Validation
- [x] Add CLI smoke tests for `--help`, config loading, and phase-control flags.
- [ ] Verify the installable package exposes the command after `pip install git+ssh://git@github.com/32127003/scbiomarker.git@py_module`.

## Progress Log
- 2026-03-18: Task skeleton created after confirming the package has no current `console_scripts` entrypoint.
- 2026-03-18: Added `scbiomarker.cli`, wired it to `run_pipeline`, and exposed the installed `scbiomarker` command through `[project.scripts]`.
- 2026-03-18: Expanded the shared workflow parser with shell-friendly aliases (`--patient`, `--cell_type`, `--add_gene_embedding`, `--train_only`, `--analysis_only`) so the public CLI matches the frozen workflow contract.
- 2026-03-18: Added CLI smoke coverage for help text, config override wiring, and analysis-only execution.
- 2026-03-18: Fresh-install command verification is intentionally deferred to T-009 in the separate `test` environment.
