# Backlog

## Dependency Order
- `T-002 -> T-003 -> T-004`
- `T-002 -> T-003 -> T-005`
- `T-002 -> T-003 -> T-005 -> T-006`
- `T-004 + T-005 + T-006 -> T-007`
- `T-007 -> T-008 -> T-009`

## TODO (top priority)
- [x] T-002: lock workflow contract, paper parity, and the user-facing interface
- [ ] T-003: introduce a generic YAML config layer with CLI override precedence

## Now
- [x] T-001: create `pyproject.toml` for pip module

## Next
- [ ] T-004: generalize split/preselection to explicit h5ad, column, and PPI inputs
- [ ] T-005: support arbitrary named multi-view gene embedding inputs
- [ ] T-006: extract reusable train/analysis phase APIs from script-style entrypoints
- [ ] T-007: add a top-level end-to-end Python pipeline API
- [ ] T-008: add a package CLI entrypoint with phase control and config overrides
- [ ] T-009: validate the installable workflow end-to-end and document it in the test project

## Parked
