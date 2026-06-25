# Feature Landscape — v0.8.0 DCM for Evoked Responses (CMC ERP + MMN demo)

**Domain:** Time-domain EEG/MEG evoked-response DCM (canonical microcircuit), forward + SPM12-parity, with a mismatch-negativity (MMN) precision-sweep demo.
**Researched:** 2026-06-25
**Overall confidence:** HIGH (grounded in SPM12 `toolbox/dcm_meeg/` source read directly + primary DCM-ERP/MMN literature)

---

## 0. How DCM for Evoked Responses works (orientation for the roadmap)

The forward pipeline SPM12 implements (and what v0.8.0 must reproduce) is, end to end:

```
stimulus input u(t)            sources × CMC populations           extrinsic coupling          observation
 (Gaussian bump,    ──►  each source = 4-population canonical  ──►  A_forward / A_backward / ──►  lead field L  ──►  scalp y(t)
  spm_erp_u)              microcircuit ODE (spm_fx_cmc):            A_lateral between sources   (spm_lx_erp /     (channels × time)
                          ss, sp, ii, dp  (8 states/source)        + B condition modulation     spm_erp_L)              │
                                                                   + C input gating                                    ▼
                                                              integrate over peristimulus time         per-condition ERP, then
                                                              (spm_int_L, spm_gen_erp)               deviant − standard = MMN
```

Verified mapping to SPM12 source (the parity reference at `../spm12/toolbox/dcm_meeg/`):

| Stage | SPM12 file | What it does |
|-------|-----------|--------------|
| Population ODE | `spm_fx_cmc.m` | 8 states/source: ss/sp/ii/dp voltage+conductance; sigmoid firing `S(V)`; synaptic τ kernels |
| Initial/steady state | `spm_x_cmc.m`, `spm_dcm_neural_x.m` | per-condition fixed point before integration |
| Input | `spm_erp_u.m` | Gaussian subcortical impulse at onset `M.ons`, dispersion `P.R` |
| Per-condition params | `spm_gen_Q.m` | applies between-trial design `X(c,:)` → B/N modulation of connections |
| Evoked integration | `spm_gen_erp.m` → `spm_int_L.m` | integrate coupled system over peristimulus time (default `dt=4ms`, `ns=128` ≈ 512 ms) |
| Lead field | `spm_lx_erp.m`, `spm_erp_L.m` | `y = L·x`; LFP / single-dipole ECD / IMG; `P.J` selects contributing states (pyramidal depolarization) |
| Priors | `spm_cmc_priors.m` | log-normal scaling priors (pE=0, pC=1) on A, B, C, G, T, R |

**The "precision" parameter is concrete and we read it in source:** in `spm_fx_cmc.m` the intrinsic gain vector `G(:,7)` is the **superficial-pyramidal self-inhibition** (`sp -> sp (-ve self)`). The first free intrinsic parameter `P.G(:,1)` maps onto it (the remap `j = [7 2 3 4 1 5 6 8 9 10]`, so `j(1)=7`), and `P.M` adds a deep-pyramidal-depolarization-dependent modulation of that same self-inhibition (`G(:,j(1)) = G(:,j(1)).*exp(-P.M*32*S(:,7))`). **This `sp→sp` self-inhibition gain is the encoding of precision** and is exactly the knob the downstream psychosis consumer needs (Adams 2013; Ranlund 2016). [REF flag A, B below]

---

## 1. Table Stakes

Must-have for a credible CMC ERP forward + MMN demo. Missing any of these = not a real ERP-DCM.

| # | Feature | Why expected | Complexity | Depends on (existing) |
|---|---------|--------------|------------|------------------------|
| T1 | **CMC population ODE (single source, 8 states)** — ss/sp/ii/dp voltage+conductance, sigmoid firing, τ kernels, SPM `exp()` log-scaling of G/T | Core generative model; everything downstream rides on it | M | ODE integrator (`utils/ode_integrator.py`), SPM `-exp` transform pattern (`neural_state.parameterize_A`) |
| T2 | **SPM12 single-source parity** for `spm_fx_cmc` dynamics on frozen fixtures | This is the project's validation contract (Phase-32 pattern) | M | Phase-32 cross-validation harness, `../spm12` MATLAB/SPM on M3 |
| T3 | **Extrinsic coupling A: forward / backward / lateral** between sources, with the CMC laminar routing (fwd `sp→ss` & `sp→dp`; bwd `dp→sp` & `dp→ii`; lateral = reciprocal-reduced) | Hierarchical message passing is the whole point of ERP-DCM | M | T1; existing `coupled_system.py` multi-node integration conventions |
| T4 | **Condition-specific modulation B** (between-trial design `X` → per-condition scaling of selected connections) | Standard vs deviant is *a B-modulation*; no B = no MMN | M | T3; `task_dcm_model.py` bilinear-B precedent |
| T5 | **Input C + Gaussian stimulus u(t)** (onset, dispersion) driving thalamic-recipient sources (A1) | Drives the evoked response; sets latency | S | T1 |
| T6 | **Evoked integration over peristimulus time** → per-source, per-condition LFP timeseries (`spm_gen_erp` analog) | Produces the actual waveform | M | T1–T5, ODE integrator |
| T7 | **SPM12 multi-source evoked parity** vs `spm_gen_erp` for a reference A/B/C | Validation contract at the network level | M | T6, M3 SPM |
| T8 | **Single-dipole-per-source lead field** → scalp waveform (`spm_lx_erp`/`spm_erp_L` ECD path, one moment per source; `P.J` = pyramidal contribution) | Turns source LFPs into the measured ERP | S–M | T6 |
| T9 | **Deviant − standard difference wave (the MMN)** as a first-class output | The headline artifact; literally the deliverable | S | T4, T6, T8 |
| T10 | **`erp_dcm_model.py` Pyro model class** wrapping forward + log-normal priors (A,B,C,G,T,R) in the repo idiom | Makes it a DCM, not just a simulator; enables inference reuse | M | `spectral_dcm_model.py` pattern, `guides.py` |
| T11 | **Superficial-pyramidal self-inhibition exposed as a named, sweepable parameter** (`G[:,7]` / `P.G[:,0]`, plus `P.M`) | This *is* precision; the demo and the consumer both need it addressable | S | T1, T10 |
| T12 | **Precision-sweep MMN demo + figure** — sweep sp self-inhibition gain → attenuated MMN | The Adams/Ranlund signature; the actinf_physics hand-off artifact | S–M | T9, T11 |

---

## 2. Differentiators

Not strictly required, but these are what make v0.8.0 scientifically useful to the downstream psychosis-MMN consumer and distinguish it from "yet another SPM clone."

| # | Feature | Value proposition | Complexity | Depends on |
|---|---------|-------------------|------------|------------|
| D1 | **VL posterior over CMC connectivity** (reuse v0.7.0 Variational Laplace) | Posterior *uncertainty* on connections/gain — SPM gives this, but in our interpretable Pyro idiom; supports the "explicit interpretable connectivity with uncertainty" core value | M | v0.7.0 VL engine, T10 |
| D2 | **Amortized flow guide for ERP-DCM** (reuse `guides/amortized_flow.py`) | Fast forward-conditioned inference; differentiates from SPM's VL-only stack | M | amortized wrappers, T10 |
| D3 | **Clean precision → MMN-amplitude transfer curve** (quantitative attenuation vs gain, not just one overlay) | Directly consumable by actinf_physics as the `(precision) → MMN amplitude` mapping; turns a figure into a function | S | T12 |
| D4 | **Parameterised precision entry point matching the consumer's variables** — adapter-friendly API mapping `(sp self-inhibition, rIFG/A1 gain, B on fwd/bwd)` to CMC params | Makes the Phase-133 actinf adapter a thin map, per the scope hand-off | S | T11, T10 |
| D5 | **BMR/model-comparison over MMN modulation hypotheses** (which connections carry the deviant effect: forward-only vs backward-only vs intrinsic-gain) | Reuses existing BMR; lets the consumer test *where* precision deficits sit (the Garrido/Ranlund model-space question) | M | existing BMR, T10 |
| D6 | **Circuit-explorer viz of the 5-source MMN network** with condition-modulated edges highlighted | Repo already has circuit-explorer; near-free interpretability win | S | `utils/circuit_viz.py`, T3/T4 |
| D7 | **MNE/BIDS-compatible ERP output object** (channels × time, montage-aware) | Repo already has MNE/BIDS IO; makes forward output drop-in for later empirical work | S | existing MNE/BIDS IO, T8 |

---

## 3. Anti-Features (explicitly NOT this milestone)

| Anti-feature | Why avoid this milestone | What to do instead |
|--------------|--------------------------|--------------------|
| **Empirical ERP data fitting** (fitting real MMN recordings) | Scope is forward + parity; the consumer runs it forward-only. Fitting adds data-IO, preprocessing, and convergence risk with no demo payoff | Validate forward vs SPM12; defer fitting to a later milestone once forward is trusted |
| **Full sensor montage / realistic head model / BEM forward** (`ft_compute_leadfield`, fieldtrip vol conductors) | Single-dipole-per-source ECD (or even LFP gain) is sufficient to produce a difference wave and the sweep; full lead fields pull in fieldtrip/coregistration | Single ECD dipole per source; `P.J` = pyramidal depolarization. Keep `spm_erp_L` ECD path minimal |
| **Source localization / inverse problem (IMG reconstruction, beamforming)** | DCM assumes sources are *given*; localization is a different problem and not needed for forward MMN | Fix source locations as priors (canonical MNI coords); no inversion of spatial model |
| **Group PEB / hierarchical between-subject modeling** | No subjects this milestone; single forward generative model | Single-model forward; BMR within-model only (D5) if at all |
| **Jansen-Rit / ERP (3-population) model** | CMC is the confirmed choice precisely because it exposes sp self-inhibition = precision; ERP model does not | CMC only. Note ERP model exists in SPM (`spm_fx_erp.m`) as future option, but do not build it |
| **CMC_2014 / thalamo-cortical / TFM variants** (`spm_fx_cmc_2014.m`, `spm_fx_cmc_tfm.m`) | Extra populations/complexity, not needed for the canonical MMN result | Classic `spm_fx_cmc.m` only |
| **Frequency-domain / steady-state (CSD) ERP** | That's the spectral path the repo already has; this milestone is *time-domain evoked* | Reuse existing spectral DCM for CSD; keep ERP time-domain |
| **Delay differential-equation full delay operator** (the `Q = spm_dcm_delay` path) | Adds Jacobian/delay machinery; SPM's `spm_gen_erp` already runs with delays off-able (`M.N`); not needed for a correct difference wave | Integrate without the full delay operator first; add only if parity demands it (flag for Phase 34 research) |

---

## 4. The MMN/oddball paradigm as DCM models it (concrete)

**Paradigm.** Auditory oddball: frequent *standard* tones vs rare *deviant* tones. Two conditions (`X` has 2 rows). The MMN is the **deviant − standard difference wave**, a fronto-central negativity peaking ~**100–250 ms** post-stimulus (P300/P3a follows later ~300 ms). [REF flag C, D]

**How the difference emerges in DCM.** Standard and deviant are *the same network* with a **condition-specific modulation `B`** of selected connections (encoded via the between-trial design `X` → `spm_gen_Q` → scaling of A/intrinsic on the deviant trial). Under predictive coding, repeated standards build a prediction (strengthened predictions / reduced PE); the deviant violates it, transiently boosting **forward (ascending, prediction-error) connections** and changing **intrinsic gain**, yielding the extra deviant response = MMN. [REF flag C, D, E]

**Which CMC parameter is "precision," and the attenuation mechanism.**
- Precision = **gain (excitability) of superficial pyramidal cells**, implemented as their **self-inhibition** `G[:,7]` (`sp→sp`, `-ve self`) in `spm_fx_cmc.m`, with the dp-dependent modulation `P.M`. Superficial pyramidal cells broadcast **prediction errors** up the hierarchy; their gain sets the *precision* (weighting) of those errors.
- **Sweep mechanism:** *increasing* sp self-inhibition ⇒ *lower* superficial-pyramidal gain ⇒ prediction errors are down-weighted ⇒ the deviant-evoked PE response shrinks ⇒ **attenuated MMN amplitude**. Conversely lower self-inhibition ⇒ higher gain ⇒ larger MMN.
- This is the **Adams 2013** "aberrant precision" account and the **Ranlund 2016** empirical finding: impaired sp self-inhibition gain in **rIFG and bilateral A1** in psychosis + relatives, with a reversed/attenuated deviant response. [REF flag A, B]

---

## 5. Canonical "forward MMN demo" deliverable (RECOV-style success criteria)

This is concrete enough to become acceptance criteria.

**5-source auditory MMN network** (Garrido et al.; Ranlund et al.): [REF flag B, C, D]

| Source | Abbrev | Approx MNI (mm) | Role |
|--------|--------|-----------------|------|
| Left primary auditory cortex | A1 L | (−42, −22, 7) | input-recipient (C drives here) |
| Right primary auditory cortex | A1 R | (46, −14, 8) | input-recipient (C drives here) |
| Left superior temporal gyrus | STG L | (−61, −32, 8) | mid-hierarchy |
| Right superior temporal gyrus | STG R | (59, −25, 8) | mid-hierarchy |
| Right inferior frontal gyrus | rIFG | (46, 20, 8) | top of hierarchy; key psychosis node |

*(coordinates are the commonly used Garrido/Ranlund values — verify exact values against the cited papers before hard-coding; flag for Zotero.)*

**Extrinsic connection graph** (hierarchy A1 → STG → IFG):
- **Forward** (ascending, `sp→ss`/`sp→dp`): A1 L→STG L, A1 R→STG R, STG R→rIFG (and commonly STG L→rIFG).
- **Backward** (descending, `dp→sp`/`dp→ii`): the reverse of each forward edge (rIFG→STG, STG→A1).
- **Lateral**: A1 L↔A1 R and STG L↔STG R (bilateral homologues).
- **Input C**: stimulus enters bilateral A1.

**Condition modulation B (deviant vs standard).** The deviant modulates a defined subset — the canonical MMN model space is "where does the deviant effect live":
- Headline demo: **forward + backward connections** modulated (full predictive-coding MMN), **plus intrinsic sp-gain** modulation at A1 and rIFG (the Ranlund winning model: intrinsic modulation in bilateral A1 + rIFG).
- This is the natural BMR model-space for differentiator D5 (forward-only vs backward-only vs intrinsic-only vs all).

**Expected waveform morphology (acceptance):**
1. Each condition produces a plausible auditory ERP (early ~50–100 ms deflection, later components), no NaN over the full peristimulus window (~0–500 ms, `dt≈4 ms`).
2. **Deviant − standard difference wave is non-zero**, negative-going, peaking ~**100–250 ms**, **larger over frontal sources** (rIFG contribution) than purely sensory — the MMN signature.
3. **Precision-sweep figure:** sweeping sp self-inhibition gain (at rIFG and/or A1) monotonically **attenuates** MMN peak amplitude; produce a `gain → |MMN|` curve (D3) plus an overlay of difference waves at low/baseline/high gain.

**Forward-parity acceptance** (the project's contract): at fixed reference (A,B,C,G,T,R), the per-source LFPs and the scalp difference wave match `spm_gen_erp` + `spm_lx_erp` within documented tolerance on frozen fixtures.

---

## 6. Feature dependency map

```
T1 CMC ODE ──► T2 single-source parity
   │
   ├─► T3 extrinsic A(fwd/bwd/lat) ──► T4 condition B ──┐
   ├─► T5 input C + u(t) ────────────────────────────────┤
   │                                                     ▼
   └────────────────────────────────► T6 evoked integration ──► T7 multi-source parity
                                                     │
                                                     ▼
                                          T8 lead field ──► T9 difference wave (MMN)
                                                     │
                            T10 Pyro model class ◄───┘
                                  │
              T11 expose sp self-inhibition ──► T12 precision-sweep demo  ◄── headline artifact
                                  │
        D1 VL · D2 amortized · D3 transfer curve · D4 adapter API · D5 BMR · D6 viz · D7 MNE output
```

## 7. MVP recommendation

Ship in priority order: **T1→T2** (parity foundation), **T3→T7** (network + evoked + parity), **T8→T9** (scalp + MMN), **T10→T12** (model class + the precision sweep). That sequence *is* the headline demo. Add **D3 + D4** (transfer curve + adapter API) because they are what the actinf_physics consumer literally imports — cheap and high-value. Defer D1/D2 (VL/amortized inference) and D5 (BMR) to "if time," since the milestone is forward-only; they are reuse-of-existing, low-risk, and can slot in without reshaping the forward stack.

---

## Sources & references (flag for manual Zotero addition — do NOT fabricate bib keys)

Primary DCM-ERP / MMN literature (add to Zotero, then cite with Better BibTeX keys):

- **[REF flag A]** Adams RA, Stephan KE, Brown HR, Frith CD, Friston KJ (2013). *The computational anatomy of psychosis.* Frontiers in Psychiatry 4:47. — aberrant precision / postsynaptic gain account. https://pubmed.ncbi.nlm.nih.gov/23750138/ (HIGH)
- **[REF flag B]** Ranlund S, Adams RA, Díez Á, et al. (2016). *Impaired prefrontal synaptic gain in people with psychosis and their relatives during the mismatch negativity.* Human Brain Mapping 37(1):351–365. — impaired sp self-inhibition gain in rIFG + bilateral A1; 5-source network; winning model = intrinsic modulation in bilateral A1 + rIFG. https://pmc.ncbi.nlm.nih.gov/articles/PMC4843949/ (HIGH)
- **[REF flag C]** Garrido MI, Kilner JM, Stephan KE, Friston KJ (2009). *The mismatch negativity: a review of underlying mechanisms.* Clinical Neurophysiology 120(3):453–463. — predictive-coding account; forward/backward modulation. (HIGH — widely cited; verify exact pages in Zotero)
- **[REF flag D]** Garrido MI, Kilner JM, Kiebel SJ, Friston KJ (2007/2009). *Dynamic causal modelling of evoked potentials: a reproducibility study* / MMN DCM studies establishing the A1 L/R, STG L/R, rIFG network with forward/backward/lateral connections. NeuroImage. (HIGH — confirm precise citation/coords in Zotero)
- **[REF flag E]** Bastos AM, Usrey WM, Adams RA, Mangun GR, Fries P, Friston KJ (2012). *Canonical microcircuits for predictive coding.* Neuron 76(4):695–711. — laminar CMC; superficial pyramidal = ascending prediction error, deep pyramidal = descending predictions. (HIGH)
- **[REF flag F]** Moran RJ, Pinotsis DA, Friston KJ (2013). *Neural masses and fields in dynamic causal modeling.* Frontiers in Computational Neuroscience 7:57. — CMC equations/parameterisation reference for the forward model. (HIGH)
- **[REF flag G]** David O, Friston KJ (2003). *A neural mass model for MEG/EEG: coupling and neuronal dynamics.* NeuroImage 20:1743–1755. — cited at the head of every `spm_fx_*`/`spm_lx_erp` file; the canonical neural-mass + lead-field reference. (HIGH)

SPM12 source (parity ground truth, read directly at `../spm12/toolbox/dcm_meeg/`):
- `spm_fx_cmc.m` (CMC ODE; `G(:,7)`=sp self-inhibition=precision; `j=[7 2 3 4 1 5 6 8 9 10]`), `spm_gen_erp.m` + `spm_int_L.m` (evoked integration), `spm_gen_Q.m` (condition modulation), `spm_erp_u.m` (Gaussian input), `spm_lx_erp.m` + `spm_erp_L.m` (lead field; LFP/ECD/IMG), `spm_cmc_priors.m` (log-normal priors). (HIGH — primary source)

> Note: MNI coordinates in §5 are the commonly cited Garrido/Ranlund values and must be verified against the primary papers before hard-coding into fixtures. Do not invent bib keys; collate the above into Zotero first.
