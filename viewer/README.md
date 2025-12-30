Viewer module plan
==================

Purpose
-------
Interactive UI to:
- select an instance and config,
- launch a solver run,
- overlay BKS vs solver routes on a plot with toggleable layers,
- browse logged stages/repo stats.

Architecture
------------
- Backend: lightweight FastAPI/Flask service exposing:
  - GET /instances
  - POST /run {instance, solver: V5, config, seed}
  - GET /solution?run_id=...&layer={solver|bks|stage}
  - GET /log?run_id=...
  - GET /analyzer?run_id=...&instance=...
- Frontend: single-page JS (Canvas/SVG, e.g., D3/vanilla) served locally. Layers:
  - Points for depot/customers; polylines for routes.
  - Toggles for BKS, solver, per-stage snapshots.
  - Sidebar for costs/load, repo bucket counts, stage deltas.

Data contracts
--------------
- Solution JSON:
  {
    "coords": {"1":[x,y], ...},
    "depot": depot_id,
    "routes": [{"id":"solver_r1","nodes":[...],"cost":123,"load":95,"layer":"solver"}, ...]
  }
- Log summary JSON:
  - stages: [{stage, cost, routes}]
  - repo: bucket counts, kept/pruned counts.
  - analyzer: gap_to_bks, coverage stats.

Notes
-----
- Keep plotting decimated for very large instances (10k): default to sampling or hulls, with on-demand full routes.
- Reuse parser/evaluator for consistent distances; rely on SQLite logs for stage snapshots.
