# Methods

## 1. Generative Model

### 1.1 Neural State Equation

The neural dynamics of the DCM are described by the bilinear state equation
([REF-001] Eq. 1; Friston, Harrison & Penny, 2003):

$$\frac{dx}{dt} = Ax + Cu(t)$$

where $x \in \mathbb{R}^N$ is the neural activity vector for $N$ brain
regions, $A \in \mathbb{R}^{N \times N}$ is the effective connectivity
matrix encoding directed causal influences between regions, $C \in
\mathbb{R}^{N \times M}$ maps $M$ experimental inputs $u(t)$ to regions,
and $t$ denotes time. The diagonal elements $a_{ii}$ represent
self-inhibition, and off-diagonal elements $a_{ij}$ represent the
influence of region $j$ on region $i$.

The A matrix is parameterized following SPM12 conventions to guarantee
stability:

$$a_{ii} = -\exp(A^{\text{free}}_{ii}) / 2, \qquad a_{ij} = A^{\text{free}}_{ij} \quad (i \neq j)$$

where $A^{\text{free}}$ denotes the unconstrained free parameters. This
ensures negative self-connections (default $-0.5$ Hz when the free
parameter is zero).

See `src/pyro_dcm/forward_models/neural_state.py` for implementation.

### 1.2 Hemodynamic Model

The Balloon-Windkessel model ([REF-002] Eq. 2-5; Stephan et al., 2007)
translates neural activity into hemodynamic responses through four
coupled ordinary differential equations per region. We implement these
in log-space following SPM12 (`spm_fx_fmri.m`) for numerical stability:

**Vasodilatory signal** ([REF-002] Eq. 2):
$$\frac{ds}{dt} = x - \kappa s - \gamma (f - 1)$$

**Blood flow** ([REF-002] Eq. 3, log-space chain rule):
$$\frac{d \ln f}{dt} = \frac{s}{f}$$

**Blood volume** ([REF-002] Eq. 4, log-space chain rule):
$$\frac{d \ln v}{dt} = \frac{f - v^{1/\alpha}}{\tau v}$$

**Deoxyhemoglobin** ([REF-002] Eq. 5, log-space chain rule):
$$\frac{d \ln q}{dt} = \frac{f E(f, E_0) / E_0 - v^{1/\alpha} q / v}{\tau q}$$

where $E(f, E_0) = 1 - (1 - E_0)^{1/f}$ is the oxygen extraction
fraction.

| Parameter | Symbol | Default | Unit | Description |
|-----------|--------|---------|------|-------------|
| Signal decay | $\kappa$ | 0.64 | s$^{-1}$ | Rate of vasodilatory signal decay |
| Autoregulation | $\gamma$ | 0.32 | s$^{-1}$ | Flow-dependent elimination |
| Transit time | $\tau$ | 2.00 | s | Hemodynamic transit time |
| Grubb's exponent | $\alpha$ | 0.32 | -- | Vessel stiffness |
| O$_2$ extraction | $E_0$ | 0.40 | -- | Resting oxygen extraction fraction |

All default values follow SPM12 code (`spm_fx_fmri.m`), not the values
in Stephan et al. (2007) Table 1.

See `src/pyro_dcm/forward_models/balloon_model.py` for implementation.

### 1.3 BOLD Signal

The BOLD percent signal change is computed from hemodynamic states via
the simplified Buxton observation equation ([REF-002] Eq. 6; Stephan
et al., 2007):

$$y = V_0 \left[ k_1 (1 - q) + k_2 \left(1 - \frac{q}{v}\right) + k_3 (1 - v) \right]$$

where:
- $k_1 = 7 E_0 = 2.8$
- $k_2 = 2.0$
- $k_3 = 2 E_0 - 0.2 = 0.6$
- $V_0 = 0.02$ (resting venous blood volume fraction)

At steady state ($v = 1, q = 1$), the BOLD signal is zero. Typical
signal change is 0.5--5% for physiological hemodynamic states.

See `src/pyro_dcm/forward_models/bold_signal.py` for implementation.

### 1.4 Spectral DCM

For resting-state fMRI, where no task stimulus is available, spectral
DCM operates in the frequency domain using cross-spectral density (CSD)
as the data representation ([REF-010]; Friston, Kahan, Biswal & Razi,
2014).

**Transfer function** ([REF-010] Eq. 3):
$$H(\omega) = C_{\text{out}} (i\omega I - A)^{-1} C_{\text{in}}$$

In the standard spectral DCM convention, $C_{\text{in}} = C_{\text{out}}
= I_N$. For numerical stability, the transfer function is computed via
eigendecomposition of $A$:

$$H(\omega) = \sum_k \frac{\mathbf{g}_k \mathbf{u}_k^T}{i 2\pi\omega - \lambda_k}$$

where $\lambda_k$ are the eigenvalues of $A$ and $\mathbf{g}_k$,
$\mathbf{u}_k$ are the corresponding output and input projections
through the eigenvectors. Eigenvalues are stabilized by clamping real
parts to $\max(-1/32)$ following the SPM12 convention for fMRI
frequencies.

**Predicted CSD** ([REF-010] Eq. 4):
$$S(\omega) = H(\omega) G_u(\omega) H(\omega)^{\dagger} + G_n(\omega)$$

where $G_u$ is the neuronal noise CSD and $G_n$ is the observation noise
CSD.

**Neuronal noise** ([REF-010] Eq. 5-6):
$$G_{u,i}(\omega) = C \cdot \exp(a_{0,i}) \cdot \omega^{-\exp(a_{1,i})} \cdot 4$$

A 1/$f^\alpha$ power-law spectrum with per-region amplitude and exponent
parameters in log-space. $C = 1/256$ is the SPM scaling constant.

**Observation noise** ([REF-010] Eq. 7):
$$G_n(\omega) = G_{\text{global}}(\omega) \cdot \mathbf{1}\mathbf{1}^T + \text{diag}(G_{\text{regional}}(\omega))$$

where the global component $G_{\text{global}}(\omega) = C \cdot
\exp(b_0) \cdot \omega^{-\exp(b_1)/2} / 8$ fills all entries, and the
regional component adds to the diagonal only. The observation noise
exponent is halved relative to neuronal noise, producing a flatter
spectrum.

Total noise parameters per model: $4N + 2$ (neuronal: $2N$, global
observation: 2, regional observation: $2N$).

See `src/pyro_dcm/forward_models/spectral_transfer.py` and
`src/pyro_dcm/forward_models/spectral_noise.py` for implementation.

### 1.5 Regression DCM

Regression DCM (rDCM; [REF-020]; Frassle et al., 2017) reformulates the
DCM as a linear regression problem in the frequency domain, enabling
analytic Bayesian inference that scales to whole-brain networks.

The forward model proceeds in three stages:

1. **HRF generation**: A hemodynamic response function is obtained by
   Euler-integrating a minimal 1-region DCM ($A = [-1], C = [16]$) with
   a unit impulse input.

2. **BOLD generation**: Given connectivity $A$, input weights $C$, and
   stimulus $u(t)$, BOLD time series are generated via Euler integration
   of the coupled neural-hemodynamic system with 3x zero-padded FFT
   convolution.

3. **Frequency-domain design matrix** ([REF-020] Eq. 4-8): For each
   region $j$, a design matrix $X_j$ is constructed in the frequency
   domain containing the DFT of HRF-convolved BOLD from other regions
   and HRF-convolved stimulus inputs. The derivative coefficients
   $c_k = \exp(2\pi i k / N) - 1$ transform the DFT to temporal
   derivatives.

The per-region regression model is ([REF-020] Eq. 5-6):
$$y_j = X_j \theta_j + \varepsilon_j$$

where $y_j$ is the DFT of region $j$'s BOLD signal (split into real and
imaginary parts), $X_j$ is the design matrix, $\theta_j$ contains the
connectivity and input parameters for region $j$, and $\varepsilon_j$ is
Gaussian noise.

See `src/pyro_dcm/forward_models/rdcm_forward.py` for implementation.

---

## 2. Inference

### 2.1 Stochastic Variational Inference

We replace the Variational Laplace (VL) algorithm used in SPM12
([REF-040]; Friston et al., 2007) with stochastic variational inference
(SVI; [REF-041]; Blei, Kucukelbir & McAuliffe, 2017) implemented in the
Pyro probabilistic programming framework ([REF-060]; Bingham et al.,
2019).

The ELBO objective is:
$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)} \left[ \log p(y, \theta | m) \right] - \mathbb{E}_{q_\phi(\theta)} \left[ \log q_\phi(\theta) \right]$$

where $q_\phi(\theta)$ is the variational distribution parameterized by
$\phi$, $p(y, \theta | m)$ is the joint probability under model $m$, and
the ELBO provides a lower bound on the log model evidence
$\log p(y | m)$.

The variational family is a mean-field Gaussian (Pyro's `AutoNormal`):
$$q_\phi(\theta) = \prod_i \mathcal{N}(\theta_i; \mu_i, \sigma_i^2)$$

Each latent variable receives independent location ($\mu_i$) and scale
($\sigma_i$) variational parameters. The initial scale is set to 0.01 to
prevent ODE blow-up from extreme initial parameter samples during
optimization.

Optimization uses the ClippedAdam optimizer with:
- Initial learning rate: 0.01
- Gradient clipping norm: 10.0
- Exponential learning rate decay to 1% of initial over the training run
- NaN detection with early termination

The SVI step sizes for ODE-based models use fixed-step rk4 integration
with $\Delta t = 0.5$ s for predictable runtime during optimization
(finer integration with adaptive solvers is used for simulation).

**Priors:**
- $A_{\text{free}} \sim \mathcal{N}(0, 1/64)$ — SPM12 convention
- $C \sim \mathcal{N}(0, 1)$ — driving input weights
- Noise parameters: variant-specific (see Sections 1.4 and 1.5)

See `src/pyro_dcm/models/guides.py` for implementation.

### 2.2 Amortized Inference

For applications requiring rapid posterior estimation across many
subjects, we provide an amortized inference approach using normalizing
flows ([REF-042]; Papamakarios et al., 2021) conditioned on summary
statistics of the observed data ([REF-043]; Cranmer, Brehmer & Louppe,
2020).

The amortized guide takes the form:
$$q_\phi(\theta | \mathbf{x}) = f_\phi(\mathbf{z}; \text{summary}(\mathbf{x}))$$

where $f_\phi$ is a Neural Spline Flow (Zuko NSF; [REF-062]) that
transforms a base distribution through monotonic rational-quadratic
spline transformations on $[-5, 5]$, and $\text{summary}(\cdot)$ is a
learned compression network.

**Summary networks:**
- **BOLD data (task DCM):** 1D convolutional neural network
  (Conv1d $\to$ ReLU $\to$ adaptive pooling $\to$ fully connected)
  producing a fixed-dimensional embedding regardless of time series
  length.
- **CSD data (spectral DCM):** Multi-layer perceptron (flatten $\to$
  FC $\to$ ReLU $\to$ FC) for the compact frequency-domain
  representation.

**Parameter packing:** Named Pyro sample sites are packed into a flat
standardized vector for the flow, with positive parameters (noise
precision, CSD noise scale) stored in log-space to enforce positivity.
Standardization to the $[-5, 5]$ range matches the NSF spline bin
domain.

**Training:** The amortized guide is trained on a dataset of simulated
observations with shuffled epoch-based optimization. Each training
example consists of (parameters, observation) pairs generated by the
simulator. After training, inference for a new subject requires only a
single forward pass through the summary network and flow.

See `src/pyro_dcm/guides/amortized_flow.py` and
`src/pyro_dcm/guides/summary_networks.py` for implementation.

### 2.3 Analytic VB for rDCM

Regression DCM admits closed-form variational Bayesian inference due to
its conjugate prior structure ([REF-020] Eq. 11-14; Frassle et al.,
2017).

**Prior specification** ([REF-020] Eq. 9-10):
- $\theta_j \sim \mathcal{N}(m_0, \Lambda_0^{-1})$ with
  $m_0^A = -0.5 I$ (self-inhibition), $m_0^C = 0$
- Precision: $\Lambda_0^{A,\text{off}} = N/8$,
  $\Lambda_0^{A,\text{diag}} = 8N$, $\Lambda_0^C = 1$ for present
  connections
- Noise precision: $\lambda_j \sim \text{Gamma}(a_0, b_0)$ with
  $a_0 = 2, b_0 = 1$

**Posterior** ([REF-020] Eq. 11-12):
$$\Sigma_j = (\Lambda_0 + \langle\lambda_j\rangle X_j^H X_j)^{-1}$$
$$\mu_j = \Sigma_j (\Lambda_0 m_0 + \langle\lambda_j\rangle X_j^H y_j)$$

**Noise precision update** ([REF-020] Eq. 13-14):
$$a_N = a_0 + N_y / 2$$
$$b_N = b_0 + \frac{1}{2}(y^H y - \mu^T \Sigma^{-1} \mu + m_0^T \Lambda_0 m_0)$$

**Free energy** ([REF-020] Eq. 15):
$$F = \sum_{r=1}^{N} F_r$$

computed as the sum of five terms per region: log-likelihood, prior
complexity, posterior entropy, noise prior, and noise posterior terms.

**Sparse variant** ([REF-021]; Frassle et al., 2018): Extends the rigid
model with Automatic Relevance Determination (ARD) using binary
indicators $z_{j,k} \sim \text{Bernoulli}(p_0 = 0.5)$ for each
connection. The sparse free energy includes two additional terms for the
binary indicator prior and posterior.

See `src/pyro_dcm/forward_models/rdcm_posterior.py` for implementation.

---

## 3. Bayesian Model Comparison

Model comparison follows the standard DCM approach of comparing the
(approximate) log model evidence across competing models.

**ELBO-based comparison (task and spectral DCM):**

For models fitted with SVI, the final ELBO serves as a lower bound on
the log model evidence. The model with the higher (less negative) ELBO
is preferred, as it provides the best balance between data fit and model
complexity (Occam's razor through the variational bound; [REF-040]).

**Free energy comparison (rDCM):**

For rDCM with analytic VB, the free energy $F$ provides an exact (within
the variational approximation) bound on the log model evidence
([REF-020] Eq. 15). The total free energy is the sum of per-region free
energies.

In both cases, a difference of $\Delta > 3$ nats (roughly corresponding
to a Bayes factor $> 20$) is conventionally considered strong evidence in
favor of the winning model.

See `src/pyro_dcm/inference/model_comparison.py` for implementation.

---

## 4. Implementation

### 4.1 Software Stack

Pyro-DCM is implemented in Python 3.11+ using the following core
libraries:

| Library | Version | Role |
|---------|---------|------|
| PyTorch | 2.x | Tensor computation and automatic differentiation |
| Pyro | 1.9+ | Probabilistic programming, SVI, ELBO computation |
| torchdiffeq | 0.2+ | ODE integration (rk4 fixed-step for SVI, dopri5 for simulation) |
| Zuko | -- | Neural Spline Flow for amortized guides |
| scipy | -- | Signal processing (Welch CSD estimation) |
| NumPyro | -- | NUTS validation of posteriors (JAX backend, optional) |
| matplotlib | -- | Diagnostic plots and benchmark figures |

### 4.2 Numerical Stability

Several measures ensure robust numerical behavior:

1. **Log-space hemodynamic states:** Blood flow, volume, and
   deoxyhemoglobin are represented as $\ln f, \ln v, \ln q$ to enforce
   positivity without explicit constraints. Log-flow is clamped at
   $\ln f \geq -14$ (i.e., $f \geq 1.2 \times 10^{-6}$) before oxygen
   extraction computation.

2. **Eigenvalue stabilization:** For spectral DCM, the real parts of
   $A$'s eigenvalues are clamped to $\max(-1/32)$ in the transfer
   function computation, preventing resonance blow-up at low frequencies.

3. **Log-space parameters in Pyro:** Positive parameters (noise
   precision, CSD noise scale) are stored in log-space in the packed
   parameter vector and exponentiated at the boundary.

4. **Gradient clipping:** The ClippedAdam optimizer clips gradient norms
   to 10.0, preventing exploding gradients from stiff ODE dynamics.

5. **NaN protection:** ODE integration outputs are checked for NaN;
   divergent BOLD predictions are detached and zero-replaced to prevent
   gradient corruption.

6. **Matrix operations:** `torch.linalg.solve` is used instead of
   explicit matrix inversion; `torch.linalg.eig` for eigendecomposition.

### 4.3 ODE Integration

Two integration strategies are used depending on context:

- **Simulation:** Adaptive dopri5 (Dormand-Prince 4/5) with discontinuity
  handling via `jump_t` / `grid_points` at stimulus onset/offset times.
  Step size hint: $\Delta t = 0.01$ s.

- **SVI optimization:** Fixed-step rk4 (fourth-order Runge-Kutta) with
  $\Delta t = 0.5$ s. The coarser step provides predictable runtime per
  SVI step (the adaptive solver's variable compute graph creates
  inconsistent optimization behavior).

---

## 5. Benchmarks

### 5.1 Protocol

Parameter recovery benchmarks follow the simulate-infer-compare protocol:

1. **Simulate:** Generate synthetic data from known ground-truth
   parameters ($A_{\text{true}}, C_{\text{true}}$) using the appropriate
   simulator (task, spectral, or rDCM).
2. **Infer:** Run inference (SVI, amortized, or analytic VB) to obtain
   posterior estimates of connectivity parameters.
3. **Compare:** Evaluate recovery accuracy using the metrics below.

All benchmarks use 3-region networks with controlled connection density
and fixed random seeds for reproducibility.

### 5.2 Metrics

| Metric | Definition | Units |
|--------|-----------|-------|
| RMSE$(A)$ | $\sqrt{\text{mean}((A_{\text{true}} - A_{\text{inferred}})^2)}$ | unitless |
| Coverage | Fraction of $A$ elements with true value inside 90% CI | proportion [0, 1] |
| Correlation | Pearson $r$ between flattened $A_{\text{true}}$ and $A_{\text{inferred}}$ | [-1, 1] |
| ELBO | Final negative ELBO loss (SVI) or free energy (rDCM) | nats |
| Wall time | Total inference time per subject | seconds |

**Amortization-specific metrics:**

| Metric | Definition |
|--------|-----------|
| RMSE ratio | RMSE(amortized) / RMSE(SVI) |
| Amortization gap | ELBO(SVI) - ELBO(amortized) |
| Speed ratio | time(amortized) / time(SVI) |

### 5.3 Results

See `docs/04_scientific_reports/benchmark_report.md` for full benchmark
results including comparison tables, ELBO traces, and posterior
visualizations.

---

## References

- [REF-001] Friston, K. J., Harrison, L., & Penny, W. (2003). Dynamic
  causal modelling. *NeuroImage*, 19(4), 1273-1302.
- [REF-002] Stephan, K. E., Weiskopf, N., Drysdale, P. M., Robinson,
  P. A., & Friston, K. J. (2007). Comparing hemodynamic models with
  DCM. *NeuroImage*, 38(3), 387-401.
- [REF-010] Friston, K. J., Kahan, J., Biswal, B., & Razi, A. (2014).
  A DCM for resting state fMRI. *NeuroImage*, 94, 396-407.
- [REF-020] Frassle, S., Lomakina, E. I., Razi, A., Friston, K. J.,
  Buhmann, J. M., & Stephan, K. E. (2017). A generative model of
  whole-brain effective connectivity. *NeuroImage*, 145, 270-275.
- [REF-021] Frassle, S., Lomakina, E. I., Kasper, L., Manjaly, Z. M.,
  Leff, A., Pruessmann, K. P., Buhmann, J. M., & Stephan, K. E. (2018).
  Regression DCM for fMRI. *NeuroImage*, 155, 406-421.
- [REF-040] Friston, K. J., Mattout, J., Trujillo-Barreto, N.,
  Ashburner, J., & Penny, W. (2007). Variational free energy and the
  Laplace approximation. *NeuroImage*, 34(1), 220-234.
- [REF-041] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017).
  Variational inference: A review for statisticians. *Journal of the
  American Statistical Association*, 112(518), 859-877.
- [REF-042] Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed,
  S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic
  modeling and inference. *Journal of Machine Learning Research*, 22(57),
  1-64.
- [REF-043] Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier
  of simulation-based inference. *Proceedings of the National Academy of
  Sciences*, 117(48), 30055-30062.
- [REF-060] Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F.,
  Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., &
  Goodman, N. D. (2019). Pyro: Deep universal probabilistic programming.
  *Journal of Machine Learning Research*, 20(28), 1-6.
