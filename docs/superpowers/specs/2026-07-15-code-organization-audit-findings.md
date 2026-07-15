# Code Organization / Refactoring Audit — Findings & Triage

**Date:** 2026-07-15
**Branch:** `gsd/phase-36-erp-dcm-model-mmn-demo`
**Method:** four parallel read-only auditors (branch quality, `src/` structure,
`scripts/`+Snakemake reproducibility, CCDS/config compliance). Static analysis
only — no edits, no compute.

Findings are grouped by theme and ranked severity × effort. Effort: S ≈ minutes,
M ≈ an hour or so, L ≈ a session. Nothing here has been changed yet; this doc is
the triage input.

---

## Theme 1 — Reproducibility blockers (the pipeline does not build the paper)

### 1.1 [HIGH · M] `generate_publication_figures.py` is wired to inputs no script produces
`scripts/generate_publication_figures.py` hard-requires six `benchmarks/results/phaseXX_*.npz`
files (`phase20_recovery`, `phase22_connectivity`, `phase23_bmr`, `phase24_foundation`,
`phase25_vae_dcm`, `phase26_sbi`; guarded at lines 256/384/497/593/667/747). **None
exist** — `benchmarks/results/` holds only `benchmark_results.json`, `recovery_matrix.*`,
`recovery_summary.txt`, `recovery_validation.csv`, and no script writes any `phaseXX_*.npz`.
The paper-figure terminal stage cannot run end-to-end. → Either restore/rewire the
`.npz` producers, or rewrite the figure loaders to read the CSV/JSON that `benchmarks/`
actually emits.

### 1.2 [HIGH · M] Cross-modal stage reads a dangling input; TRIBE chain has an external origin
`scripts/24_compare_crossmodal.py:19,195` requires `results/phase24_meeg/meeg_dcm_results.npz`,
which **no script produces**. `scripts/24_extract_tribe_latents.py` consumes an external
"TRIBE" model checkpoint that lives outside the repo. → Document both as external
pre-existing inputs (a `scripts/README` / `data/README`); for a future Snakefile they
must be declared `input:` provenance, not rules.

### 1.3 [HIGH · L] Stage-number prefixes are phase tags, not DAG order
Only `22_` and `24_` prefixes exist (plus `23_` in cluster). Numbers are milestone IDs,
not execution order; three `22_*` scripts span two different chains; 14/20 scripts have
no number. A newcomer cannot infer run order from filenames. → Either commit to true
DAG ordering or drop numeric prefixes and let a Snakefile encode order — don't leave the
half-numbered state.

---

## Theme 2 — CCDS / config centralization

### 2.1 [HIGH · L] `config.py` centralizes almost nothing
Defines only `PROJECT_ROOT`, `BENCHMARK_{RESULTS,FIGURES,FIXTURES}_DIR`, `TAPAS_RDCM_PATH`,
`MATLAB_PATH`. Missing the CCDS v2 set: `RAW/INTERIM/PROCESSED_DATA_DIR`, `MODELS_DIR`/
`CHECKPOINTS_DIR`, `FIGURES_DIR`, `RESULTS_DIR`, `REPORTS_DIR`, `CLUSTER_RESULTS_DIR`.
Only ~7 files import config at all. → **`retrofit-config` skill is a direct match.**

### 2.2 [HIGH · M] ~30 hardcoded path literals bypass `config.py`
Full table in the CCDS audit. Worst offenders:
- **Live drift bug:** `validation/run_validation.py:58`, `run_rdcm_validation.py:78,237`
  re-hardcode a MATLAB path that `config.MATLAB_PATH` already defines.
- `benchmarks/calibration_analysis.py:486,495`, `calibration_sweep.py:698`,
  `generate_fixtures.py:750`, `run_all_benchmarks.py:273`, and
  `generate_publication_figures.py:937` ignore the `BENCHMARK_*_DIR` constants **that
  already exist**.
- `scripts/22_*`, `24_*`, `train_*`, `generate_training_data.py`, `eval_hybrid_vae_dcm.py`
  hardcode `results/…`, `data/training/`, `checkpoints/`, `figures/` in argparse defaults.
- `cluster/scripts/*` — 14 sites hardcode `Path("cluster/results")`.

### 2.3 [MED · S] `cluster/logs/` and `cluster/results/*.json` are tracked in git
`git check-ignore` returns nothing for them; dozens of `*.out/*.err` and result JSON are
committed/untracked-noise. → Add `cluster/logs/` and `cluster/results/*.json` to
`.gitignore`, then `git rm -r --cached`. (`.mypy_cache/`, `.ruff_cache/`, `__pycache__/`,
`sbi-logs/`, `*.stackdump` are already correctly ignored.)

### 2.4 [MED · M] Config constant names are ad-hoc, not CCDS v2; no data split
`BENCHMARK_RESULTS_DIR` etc. are project-ad-hoc; `benchmarks/config.py` fields
`output_dir`/`figure_dir` are old-style. `data/` has only `test_gen/` — no
`raw/interim/processed/external` split. Two `config.py` files (root paths vs
`benchmarks/` dataclass) is a filename collision.

---

## Theme 3 — Phase-36 branch code quality

### 3.1 [MED · M] Triplicate copy-paste across the three network-builder modules
`forward_models/mmn_reference.py`, `collision_reference.py`, `collision_3node_reference.py`.
`_edge_mask` is **byte-identical** in all three; `_collision_scalars` identical between the
two collision modules; `_VALID_FLAGS` redefined 3×; `build_*`/`*_cmc_params` bodies
near-identical modulo hard-coded edge tuples. ~200 duplicated lines the modules' own
docstrings warn about. → Extract one topology-parametrized helper the three thin wrappers
call.

### 3.2 [MED · S] Silent NaN→zeros clamp feeds the VL finite-difference Jacobian
`inference/forward_models.py:955-956` (`ERPDCMForward.predict`) clamps a diverged trajectory
to `zeros_like` with no warning/counter. Fine as an SVI penalty idiom, but this forward is
**also** called by the VL FD-Jacobian, where a zero-clamped parameter-independent output
looks like a zero-gradient region and can mislead the optimizer. → Emit `warnings.warn` /
increment a diagnostic counter when the clamp fires (the one more-than-cosmetic item).

### 3.3 [MED · M] `erp_dcm_model` function body is ~109 lines (2× the 50-line hard limit)
`models/erp_dcm_model.py:150-259`. Eight repetitive `pyro.sample` log-space blocks. → Factor
`_sample_logspace(name, shape, var, event_dim)` + `_apply_masks(...)` helpers.

### 3.4 [LOW · S] `-32.0` dead-edge constant triplicated
`inference/forward_models.py:690`, `models/erp_dcm_model.py:52`, and `_MS_A_DEAD` in
`validation/export_to_mat.py`. → Single source of truth (e.g. `cmc_priors`).

### 3.5 [LOW · S] New CMC/ERP math cites SPM source-lines, not `REF-xxx` ids
Deliberate, well-documented deviation (Zotero/.bib discipline; `erp_leadfield.py:28-29`
explicitly says don't fabricate keys). Non-urgent: add `REF-xxx` once papers confirmed in
Zotero.

### 3.6 [LOW · S] `B_PRIOR_VARIANCE = 1/8` is a provisional unverified prior
`models/erp_dcm_model.py:54-59`, self-flagged MUST-VERIFY against `spm_cmc_priors.m`.
Low-stakes for the fixed-B demo; matters for SVI B-recovery.

**No genuine logic/shape bugs found. Test quality is strong** — real asserts, legitimately
guarded skips, no tautologies.

---

## Theme 4 — `src/` structure

### 4.1 [MED · S] Naming collision: `inference/forward_models.py` module vs `forward_models/` package
The module (1007 lines) is not a duplicate — it holds the VL `ForwardModel` Protocol +
spectral/task/ERP adapters that wrap the package. But the twin name is a readability trap.
→ Rename to `vl_forward_models.py` (update `inference/__init__.py` + 4 test imports).

### 4.2 [MED · S] Two `latent_extraction.py` with different semantics
`rnn/latent_extraction.py` (CT-RNN trajectory + PCA) vs `neural_data_models/latent_extraction.py`
(LSTM-AE latents + CSD). Zero symbol overlap; the shared filename forces disambiguation on
every read. → Rename to `trajectory_pca.py` and `latent_csd.py`.

### 4.3 [MED · M] Six modules > 700 lines, four > 1000
`variational_laplace.py` 1374, `task_simulator.py` 1058, `rdcm_posterior.py` 1028,
`inference/forward_models.py` 1007, `parameter_packing.py` 843, `models/guides.py` 722.
→ Split the >1000 first; `inference/forward_models.py` cleanly separates into three
per-variant adapter files.

### 4.4 [MED · S] `sbi_*` modules orphaned from the package `__init__`
`inference/sbi_spectral.py`, `sbi_embedding.py`, `sbi_diagnostics.py` are absent from
`inference/__init__`, used only via scripts/tests (0 src importers). → Add to `__all__` if
public.

### 4.5 [LOW · L] Pre-ERP RNN / latent-circuit stratum (~3200 lines) superseded but live
`rnn/`, `neural_data_models/`, `models/latent_circuit_dcm_model.py`,
`simulators/latent_circuit_simulator.py`, `models/hybrid_vae_dcm.py`. Disconnected from the
root public API (task/spectral/rDCM only) but **fully tested** — not deletable. → Owner
decision: promote to public API, or relocate under `experimental/`/`legacy/`.

### 4.6 [LOW · S] `*_reference.py` name misleads — these are production builders
The three network builders are the single source of truth for MMN/collision topologies, not
throwaway references. → Rename `*_reference.py` → `*_network.py`.

### 4.7 [LOW · S] Import smells
Self-import in `models/latent_circuit_dcm_model.py`; duplicated import lines in
`inference/forward_models.py` and `simulators/latent_circuit_simulator.py`.

---

## Theme 5 — cluster/ organization

### 5.1 [MED · M] `scripts/` vs `cluster/scripts/` split-brain
Real scientific compute is split by *where it runs*, not *what it does*
(`train_rnn_seed.py` ↔ `cluster/scripts/train_rnn_ensemble.py`; ERP cross-validation lives
only under `cluster/scripts/`). → Move analysis logic into `src/pyro_dcm/` (importable,
testable); make both script trees thin CLI shims.

### 5.2 [MED · S] Three sbatch conventions + orphaned Phase-16 machinery
`.sbatch` (19, all use `cluster/lib/cluster_env.sh`) vs root `.slurm` (Phase-16, 507 lines)
vs loose `.sh`. Three `erp_cross_validation_*.sbatch` differ only by a `--mode` flag. →
Converge on `.sbatch`, collapse the ERP trio into one parameterized job, archive Phase-16,
rewrite `cluster/README.md` for the current ERP workflow.

---

## Snakemake verdict: **after cleanup, then worth it**

~11 scripts already have clean argparse file-in/file-out contracts (Snakemake-ready today).
Two blockers first: (1) fix the broken figure stage (1.1), (2) make `config.py` the real
path source (2.1/2.2). Then `scaffold-cluster` collapses the sbatch sprawl and gives the
fMRI→SBI→figures path genuine end-to-end reproducibility. The `cluster_env.sh` + `.sbatch`
convention already mirrors a Snakemake SLURM profile.

---

## Execution status (2026-07-16)

Done & committed (cluster-verified where a test path exists):

- **B1** `538e08b` — untracked `cluster/logs/` + job-stamped result JSON.
- **B3.1** `6dd5ff8` — dedup 3 CMC builders → shared `_cmc_network`.
- **B3** `a8fdbc2` — unify `-32.0` constant, observable NaN clamp, factor
  `erp_dcm_model` body. (M3 job 58328297: 67 passed.)
- **B4** `2a7a578` — renames `vl_forward_models`, `latent_csd`, `trajectory_pca`;
  `sbi_*` doc note. (`pytest --collect-only`: 909 collected, no import errors.
  `*_reference→*_network` (4.6) deliberately deferred — cosmetic, files just
  restructured. Flagged import "smells" (4.7) were false positives — docstring
  doctests + intentional per-method lazy imports.)
- **B2 config.py** `527a1fc` — CCDS-v2-style constants for the repo's actual dirs.
- **B2 group A** `5f532ef` — MATLAB drift fix + benchmark/figure scripts use the
  existing constants. (`--help` smokes pass.)
- **B2 group C** `b0fe58a` — 12 `cluster/scripts` result paths → `CLUSTER_RESULTS_DIR`.
  (py_compile clean; `lc_calibration_aggregate` runs end-to-end.)

**Deferred — B2 group B (`scripts/` argparse defaults, ~13 sites).** Attempted;
stopped on a concrete obstacle. None of the 12 target scripts import config or
bootstrap `sys.path`, so each needs a `sys.path.insert(...)` added before its
imports — which triggers ruff **E402** on every subsequent third-party/pyro
import, requiring scattered `# noqa: E402`. That degrades the import hygiene of
12 *untested* scripts to centralize defaults that already resolve correctly from
the repo root. Judged not worth the mess on autopilot. Captured as a follow-up
(exact site list + the E402 caveat). A ready-but-unrun draft lives in the session
scratchpad (`migrate_scripts_config.py`, has the E402 gap).

Also flagged separately: pre-existing task-DCM test rot in
`test_vl_forward_model_protocol.py` (5 tests fail on `simulate_task_dcm`/
`make_block_stimulus` signature drift — unrelated to this refactor).

**Left as originally scoped (bigger, separate efforts):** B5 file splits,
B6 reproducibility (broken figure stage), B7 Snakemake, B8 RNN/latent-circuit
relocation.

## Proposed execution batches (independently committable)

| # | Batch | Effort | Risk | Notes |
|---|-------|--------|------|-------|
| B1 | Repo hygiene — gitignore `cluster/logs/` + `cluster/results/*.json`, `git rm --cached`, drop smoke/scratch dirs | S | none | no code behavior change |
| B2 | Config retrofit (`retrofit-config`) — CCDS v2 constants + migrate ~30 literals + fix MATLAB drift bug | M | low | prerequisite for Snakemake |
| B3 | Phase-36 branch cleanup — dedup 3 builders (3.1), NaN-clamp observability (3.2), factor `erp_dcm_model` (3.3), unify `-32.0` (3.4) | M | med | touches live branch; highest value while fresh |
| B4 | `src/` renames — `vl_forward_models`, two `latent_extraction`, `*_network`; add `sbi_*` to `__init__`; import smells | S–M | low | mechanical but ripples to imports/tests |
| B5 | File splits — the four >1000-line modules | M–L | med | churn; start with `inference/forward_models.py` |
| B6 | Reproducibility fixes — repair figure stage (1.1), document TRIBE/MEEG external deps (1.2) | M–L | med | unblocks paper build |
| B7 | Snakemake DAG (`scaffold-cluster`) — after B2/B6 | L | med | infra decision |
| B8 | RNN/latent-circuit relocation (4.5) + cluster consolidation (5.x) | L | med | owner decision required |

**Verification note:** the full pytest suite is large — per the cluster-routing rule
(>3 min → M3), full-suite runs route to M3 via sbatch; targeted fast unit tests for a given
change run on laptop.
