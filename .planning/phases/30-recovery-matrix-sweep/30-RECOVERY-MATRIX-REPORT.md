# Phase 30 Recovery Matrix Report

**Verdict summary:** 5 PASS · 1 IDENTIFIABILITY-LIMIT-WITH-EVIDENCE · 4 ERRORED (surfaced below, no silent failures). Total cells: 10.

Every cell receives an explicit verdict (VLREC-04): a cell either meets the documented per-cell thresholds (PASS) or is documented as an identifiability limit WITH evidence; error-status cells are listed explicitly, never dropped.

## Per-cell verdicts

| cell | variant | N | SNR | RMSE_A med | RMSE_A IQR | sign(masked) | cov95 | shrink | R2/region | conv | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | spectral | 2 | 1 | 0.001401 | 0.0002216–0.002454 | 1 | 1 | 0.2109 | — | 1 | PASS |
| 1 | spectral | 2 | 3 | 0.01293 | 0.008822–0.01874 | 1 | 1 | 0.2082 | — | 1 | PASS |
| 2 | spectral | 4 | 1 | 0.0007489 | 0.0005442–0.0009596 | 1 | 1 | 0.03181 | — | 1 | PASS |
| 3 | spectral | 4 | 3 | 0.0009865 | 0.0004575–0.0015 | 1 | 1 | 0.02902 | — | 1 | PASS |
| 4 | task | 2 | 1 | — | —–— | — | — | — | — | — | ERROR |
| 5 | task | 2 | 3 | — | —–— | — | — | — | — | — | ERROR |
| 6 | task | 4 | 1 | — | —–— | — | — | — | — | — | ERROR |
| 7 | task | 4 | 3 | — | —–— | — | — | — | — | — | ERROR |
| 8 | latent_circuit | 4 | 1 | 0.05011 | 0.04213–0.08192 | 1 | — | 0.502 | 0.07605 | 1 | IDENT-LIMIT |
| 9 | latent_circuit | 4 | 3 | 0.03963 | 0.03761–0.06428 | 1 | — | 0.3465 | 0.3455 | 1 | PASS |

## Identifiability limits (with evidence)

- **cell 8 (latent_circuit N=4 SNR=1.0)** — identifiability limit: failed checks rmse_a(observed=0.0501132010042016, threshold=0.05); skipped (metric None): coverage_95

## No silent failures — errored cells

These cells errored during the fit and are surfaced here (NOT dropped from the matrix); each appears in the CSV with status `error`:

- **cell 4 (task N=2 SNR=1.0)** — `underflow in dt 0.0`
- **cell 5 (task N=2 SNR=3.0)** — `underflow in dt 0.0`
- **cell 6 (task N=4 SNR=1.0)** — `underflow in dt 0.0`
- **cell 7 (task N=4 SNR=3.0)** — `underflow in dt 0.0`

## eig_clamp / stability-boundary regime (VLROBUST-03)

Ground-truth `A` was drawn with max-real-eigenvalue EXCLUDED from the near-stability-boundary band `[-0.05, 0.0]` (eig_clamp non-injectivity, pitfall N2). Accepted draws still falling in that band (should be 0): **0**.

| cell | variant | N | SNR | max-eig (max) | max-eig (min) | in-band | shrink | cov95 |
|---|---|---|---|---|---|---|---|---|
| 0 | spectral | 2 | 1 | -0.2978 | -0.5 | 0 | 0.2109 | 1 |
| 1 | spectral | 2 | 3 | -0.2978 | -0.5 | 0 | 0.2082 | 1 |
| 2 | spectral | 4 | 1 | -0.2897 | -0.4331 | 0 | 0.03181 | 1 |
| 3 | spectral | 4 | 3 | -0.2897 | -0.4331 | 0 | 0.02902 | 1 |
| 8 | latent_circuit | 4 | 1 | -0.5 | -0.5 | 0 | 0.502 | — |
| 9 | latent_circuit | 4 | 3 | -0.5 | -0.5 | 0 | 0.3465 | — |

**Overconfident / non-injective regime.** Shrinkage soft target is 0.7 (RECOV-07); very-low shrinkage is the EXPECTED Laplace overconfidence signal (job 55772525), documented here as evidence, NOT flagged as a bug. Cells with shrinkage below the soft target:

- cell 0 (spectral N=2 SNR=1.0) — shrinkage 0.2109 (< 0.7)
- cell 1 (spectral N=2 SNR=3.0) — shrinkage 0.2082 (< 0.7)
- cell 2 (spectral N=4 SNR=1.0) — shrinkage 0.03181 (< 0.7)
- cell 3 (spectral N=4 SNR=3.0) — shrinkage 0.02902 (< 0.7)
- cell 8 (latent_circuit N=4 SNR=1.0) — shrinkage 0.502 (< 0.7)
- cell 9 (latent_circuit N=4 SNR=3.0) — shrinkage 0.3465 (< 0.7)

## Interpretation

- **Thresholds are provisional.** `RMSE_A_THRESHOLD = 0.05` is the documented-default from the v0.7.0 threshold-research note (no principled Fisher-information bound exists yet); cells near the line are flagged for audit rather than treated as a hard derived tolerance.
- **Identifiability limits carry their evidence** (shrinkage, coverage, RMSE IQR, convergence) in `recovery_matrix.json`, so a marginal miss is documented, not hidden (VLREC-04).
- **eig_clamp regime held**: every accepted ground-truth draw sat outside the near-boundary band, so recovery is not confounded by eig_clamp non-injectivity; the low shrinkage at high SNR is the expected Laplace-overconfidence regime (VLROBUST-03), reported as evidence.
