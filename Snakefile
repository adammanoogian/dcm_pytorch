# =============================================================================
# Snakefile -- Test orchestration for Pyro-DCM on M3 SLURM
# =============================================================================
# Usage (on M3):
#   snakemake --profile cluster/snakemake-profile/ test_phase20_wave1
#   snakemake --profile cluster/snakemake-profile/ test_phase20
#   snakemake -n test_phase20  # dry-run
#
# Usage (from local via SSH):
#   ssh m3 "cd ~/fc37/adam/projects/dcm_pytorch && snakemake --profile cluster/snakemake-profile/ test_phase20_wave1"
# =============================================================================


# =============================================================================
# Phase 20: Latent Circuit Forward Model
# =============================================================================

rule test_latent_circuit_forward:
    """Phase 20-01: Hemodynamic toggle + latent circuit simulator tests."""
    output:
        touch("cluster/results/.test_latent_circuit_forward.done"),
    log:
        "cluster/logs/snk_latent_circuit_forward.log",
    shell:
        "python -m pytest tests/test_latent_circuit_forward.py -x -v 2>&1 | tee {log}"


rule test_multi_start_svi:
    """Phase 20-02: Multi-start SVI extension tests."""
    output:
        touch("cluster/results/.test_multi_start_svi.done"),
    log:
        "cluster/logs/snk_multi_start_svi.log",
    shell:
        "python -m pytest tests/test_multi_start_svi.py -x -v 2>&1 | tee {log}"


rule test_latent_circuit_model:
    """Phase 20-03: Pyro latent circuit DCM model + guide auto-discovery."""
    input:
        "cluster/results/.test_latent_circuit_forward.done",
    output:
        touch("cluster/results/.test_latent_circuit_model.done"),
    log:
        "cluster/logs/snk_latent_circuit_model.log",
    shell:
        "python -m pytest tests/test_latent_circuit_model.py -x -v 2>&1 | tee {log}"


rule test_latent_circuit_recovery:
    """Phase 20-04: Recovery benchmark runner tests."""
    input:
        "cluster/results/.test_latent_circuit_model.done",
    output:
        touch("cluster/results/.test_latent_circuit_recovery.done"),
    log:
        "cluster/logs/snk_latent_circuit_recovery.log",
    shell:
        "python -m pytest tests/test_latent_circuit_recovery.py -x -v 2>&1 | tee {log}"


rule test_existing_no_regression:
    """Verify existing tests pass (no regressions from Phase 20 changes)."""
    output:
        touch("cluster/results/.test_existing_no_regression.done"),
    log:
        "cluster/logs/snk_existing_regression.log",
    shell:
        "python -m pytest tests/test_neural_state.py tests/test_balloon.py "
        "tests/test_pyro_models.py -x -q 2>&1 | tee {log}"


# =============================================================================
# Aggregate targets
# =============================================================================

rule test_phase20_wave1:
    """All Wave 1 tests (plans 20-01, 20-02) + regression check."""
    input:
        "cluster/results/.test_latent_circuit_forward.done",
        "cluster/results/.test_multi_start_svi.done",
        "cluster/results/.test_existing_no_regression.done",


rule test_phase20:
    """All Phase 20 tests in dependency order."""
    input:
        "cluster/results/.test_latent_circuit_forward.done",
        "cluster/results/.test_multi_start_svi.done",
        "cluster/results/.test_latent_circuit_model.done",
        "cluster/results/.test_existing_no_regression.done",
