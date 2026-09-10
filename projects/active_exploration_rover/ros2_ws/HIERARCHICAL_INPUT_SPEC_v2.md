# Hierarchical onboard input specification v2 — 2026-09-10

Identifier: `hierarchical-onboard-v2`. Future runs only. This is a semantic
clarification supplement to the unchanged hierarchical benchmark contract v2,
not a new navigation/evaluation contract.

The compatible model-visible candidate schema remains `{id, x_m, y_m, cells}`.
`cells` means `frontier_component_cell_count`:

> frontier_component_cell_count is the number of cells in the frontier component. It is not free-space area, accessible-area coverage, or a guarantee of traversability.

The count is the size of the associated online frontier component. If multiple
components map to one approach cell, existing candidate construction retains the
largest component's size, not the sum. IDs remain snapshot-specific; positions
remain meters in the online map frame. All existing candidate filters, ranking,
route planning, local recovery, safety, model/settings and evaluator are unchanged.
No field is renamed, added to candidates, or removed. No route-cost field is added.

The definition and version are appended to the existing destination-selector
base prompt in `scripts/hierarchical_driver.py`. `driver_configuration.json` now
records `input_spec_version`; full submitted base instructions remain in RPC logs.
The historical unversioned input is treated as v1, preserved in the successful
baseline's source/scripts/hierarchical_driver.py, driver_rpc.jsonl and exact
observations. Historical contracts, prompts, logs and results are not rewritten.

Baseline: `experiment_runs/hierarchical-luna-restart-20260910/`.
Exact offline change/evidence: `experiment_runs/offline-luna-decisions-20260910/`.
No live validation or model calls accompanied this clarification.
