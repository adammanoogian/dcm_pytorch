---
phase: 30-recovery-matrix-sweep
plan: 02
subsystem: infra
tags: [variational-laplace, recovery-sweep, slurm, m3-cluster, benchmark, vl]

# Dependency graph
requires:
  - phase: 30-01
    provides: "assemble_cell_metrics + exclude_near_boundary_A/resample_A_until_accepted + snr_for_model + compute_shrinkage_ratio"
  - phase: 29-04
    provides: "method=vl runners (spectral/task/latent) + RUNNER_REGISTRY"
provides:
  - "benchmarks/recovery_matrix_grid.py — GRID constants, enumerate_cells/cell_for_index, run_one_cell driver"
  - "cluster/scripts/recovery_matrix_cell.py — env-driven single-cell SLURM entrypoint emitting per-cell JSON"
  - "cluster/sbatch/recovery_matrix_sweep.sbatch — SLURM array (0-9) over the validation grid, no-pip, dt>=0.1"
  - "Submitted M3 array job 56346424 (120 fits) — results land in cluster/results/recovery_matrix_56346424_<0..9>.json"
affects: ["30-03 (harvest + per-cell pass/identifiability classifier), 31 (BMR tempering calibrated against this coverage output)"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-cell SLURM array task runs all seeds inline (config.n_datasets), not one seed per task"
    - "Grid driver inlines the Phase 29 VL simulate->fit loop to inject per-cell SNR without forking runners"
    - "latent_circuit N-axis collapsed to fixed N=4 (no fabricated N=2 cell)"

key-files:
  created:
    - benchmarks/recovery_matrix_grid.py
    - cluster/scripts/recovery_matrix_cell.py
    - cluster/sbatch/recovery_matrix_sweep.sbatch
  modified: []

key-decisions:
  - "[30-02-D1] Inline the per-variant VL loop in the grid driver (reusing the Phase 29 simulate/forward symbols) to inject SNR, rather than forking the runners or using env-var globals."
  - "[30-02-D2] Seeds run inside one array task (10 cells -> 10 array tasks -> 120 fits), not one task per (cell,seed)."
  - "[30-02-D3] latent_circuit N-axis collapses to fixed N=4; grid emits 10 cells (spectral 4 + task 4 + latent 2)."

patterns-established:
  - "Mutagen-deployed sweep: verify files + models/ on M3, sanity-import in env, sbatch, then STOP (async harvest is the next plan)"

# Metrics
duration: 22min
completed: 2026-06-10
---

# Phase 30 Plan 02: Recovery-Matrix Sweep Driver + M3 Submission Summary

**Recovery-matrix grid driver (10 cells: N{2,4} x SNR{1,3} x {spectral,task,latent}) + env-driven SLURM single-cell entrypoint + array sbatch; local <3min faithfulness pre-check passed and the 120-fit validation grid SUBMITTED to M3 as array job 56346424 (all 10 tasks RUNNING).**

## Performance

- **Duration:** 22 min
- **Started:** 2026-06-10T20:59:53Z
- **Completed:** 2026-06-10T21:22:19Z
- **Tasks:** 2 code tasks + 1 submit checkpoint (pre-approved)
- **Files created:** 3

## Accomplishments

- `benchmarks/recovery_matrix_grid.py`: `GRID_VARIANTS/GRID_N/GRID_SNR/GRID_SEEDS`
  constants; `enumerate_cells()` / `cell_for_index()` producing a stable 10-cell
  ordering (spectral 4 + task 4 + latent_circuit 2) with the latent N-axis
  asymmetry handled; `run_one_cell()` reusing the Phase 29 VL simulate/forward
  symbols per variant, injecting per-cell SNR via `snr_for_model`, excluding
  near-boundary `A` (spectral/task), computing per-region (NOT pooled) trajectory
  R2 + shrinkage for latent, and assembling via `assemble_cell_metrics` (30-01).
- `cluster/scripts/recovery_matrix_cell.py`: env-driven entrypoint
  (`SLURM_ARRAY_TASK_ID` -> `cell_for_index` -> `run_one_cell`; env knobs
  `RECOVERY_MAX_ITER/BASE_SEED/QUICK`) with a try/except recording failures as
  `status="error"` so one bad cell never aborts the array; writes
  `cluster/results/recovery_matrix_<jobid>_<taskid>.json`.
- `cluster/sbatch/recovery_matrix_sweep.sbatch`: `--array=0-9`, `comp` partition,
  8h walltime, 16G, 4 cpus; sources `cluster_env.sh`; LOUD no-pip warning (no
  real pip command); dt>=0.1 enforced inside the driver.
- **Local faithfulness pre-check PASSED:** task 0 (spectral N=2 SNR=1, quick,
  max_iter=6) ran 10/10 seeds in **28.7s** (<3 min) and wrote a valid
  `recovery_matrix_local_0.json` with `status="ok"` and a populated `metrics`
  block (rmse_a median 0.0022, masked sign recovery 1.0, coverage 1.0,
  shrinkage 0.21).
- **M3 array job 56346424 SUBMITTED** — all 10 tasks confirmed RUNNING via
  `squeue` (including latent-circuit cells 8/9, which require the `models/`
  Mutagen sync that is confirmed in place on M3).

## Task Commits

1. **Task 1: Grid definition + single-cell driver** - `5c7547a` (feat)
2. **Task 2: Cluster entrypoint + SLURM array sbatch** - `10918ce` (feat)

**Plan metadata:** (docs commit, this SUMMARY + STATE + ROADMAP)

## Files Created/Modified

- `benchmarks/recovery_matrix_grid.py` - Grid constants, cell enumeration/index
  mapping, and the `run_one_cell` per-cell VL driver (SNR + boundary + metric
  wiring over the Phase 29 fit logic).
- `cluster/scripts/recovery_matrix_cell.py` - Env-driven single-cell SLURM
  entrypoint writing per-cell JSON.
- `cluster/sbatch/recovery_matrix_sweep.sbatch` - SLURM array job over the grid
  (no pip, dt>=0.1, sources cluster_env.sh).

## M3 Submission Record

- **Job id:** `56346424` (array `56346424_0` .. `56346424_9`)
- **Grid:** N in {2,4} x SNR in {1,3} x {spectral, task, latent_circuit},
  GRID_SEEDS=10. latent_circuit fixed N=4. = **10 cells x 10 seeds = 120 fits.**
- **Cell -> array-index mapping** (from `enumerate_cells()`):

  | idx | variant | N | SNR |
  |-----|---------|---|-----|
  | 0 | spectral | 2 | 1.0 |
  | 1 | spectral | 2 | 3.0 |
  | 2 | spectral | 4 | 1.0 |
  | 3 | spectral | 4 | 3.0 |
  | 4 | task | 2 | 1.0 |
  | 5 | task | 2 | 3.0 |
  | 6 | task | 4 | 1.0 |
  | 7 | task | 4 | 3.0 |
  | 8 | latent_circuit | 4 | 1.0 |
  | 9 | latent_circuit | 4 | 3.0 |

- **Partition / resources:** `comp`, `--time=08:00:00`, `--mem=16G`,
  `--cpus-per-task=4`.
- **Expected walltime:** spectral cells finish in minutes; task and especially
  latent-circuit cells (per-seed ODE + finite-difference Jacobian x 10 seeds) are
  the long pole — budgeted at 8h/cell, expected to complete well inside that.
- **Where results land:** `cluster/results/recovery_matrix_56346424_<0..9>.json`
  on M3, synced back to the laptop via Mutagen. Logs:
  `cluster/logs/recov_matrix_56346424_<0..9>.{out,err}`.
- **Status at submission:** all 10 array tasks RUNNING (verified via `squeue`).

## Decisions Made

- **[30-02-D1]** The grid driver INLINES the per-variant VL simulate->fit loop
  (importing the same simulate/forward symbols the Phase 29 runners use) to
  thread the per-cell SNR, rather than forking the runners or using env-var
  globals. This was the plan's preferred "no globals" seam; it keeps the SNR axis
  comparable while reusing all fit logic.
- **[30-02-D2]** Seeds run INSIDE one array task (`config.n_datasets=GRID_SEEDS`),
  so 10 cells = 10 array tasks = 120 fits. Seeds are not separate tasks (mirrors
  the existing runner per-seed loop).
- **[30-02-D3]** `latent_circuit` collapses the N axis to fixed N=4 (its ground
  truth is the fixed bilinear topology); the grid emits 10 cells, not 12, and
  never fabricates a non-existent N=2 latent-circuit cell.

## Deviations from Plan

None - plan executed exactly as written. The submit checkpoint was pre-approved
by the user (small validation grid explicitly chosen; cluster use authorized for
jobs >3 min), so it was treated as APPROVED and executed autonomously after the
local pre-check passed.

## Issues Encountered

None. The local pre-check passed first try; M3 SSH, Mutagen sync (including
`src/pyro_dcm/models/`), and the M3-env import sanity check all succeeded before
submission.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **Plan 30-03 is the harvest half** and runs AFTER job 56346424 completes:
  collect the 10 per-cell JSONs, apply documented per-cell thresholds + the
  pass/identifiability-limit classifier, build the matrix CSV/JSON, characterize
  the eig_clamp/boundary regime, and write the report (VLREC-04, VLROBUST-03).
- **Monitor:** `ssh m3 "squeue -u aman0087 --name=recov_matrix"`; when the array
  is done, results sync back via Mutagen to `cluster/results/`.
- **No blockers.** The Mutagen `models/` anchored-ignore fix is confirmed in
  place (latent-circuit cells 8/9 are running on M3).

---
*Phase: 30-recovery-matrix-sweep*
*Completed: 2026-06-10*
