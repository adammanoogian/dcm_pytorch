#!/bin/bash
# =============================================================================
# Sharded pytest runner (DCCN) -- splits a test selection across array tasks
# =============================================================================
# Use this for any suite too slow to run serially. The `slow` suite is the
# motivating case: 48 tests at ~26 min each is ~20 h serially, which does not
# fit a sane walltime. Sharded over 8 array tasks it is ~2.5 h wall.
#
# Each array task collects the SAME deterministic test list, then runs the
# subset where (index % NSHARDS) == SHARD_ID. Round-robin, not contiguous
# blocks, because adjacent tests in a file tend to have similar cost -- striping
# spreads the expensive ones across shards.
#
# Usage:
#   sbatch --array=0-7 --export=ALL,PYTEST_MARK='slow and not spm and not tapas' \
#          cluster/sbatch/pytest_sharded.sbatch
#
#   sbatch --array=0-3 --export=ALL,TEST_TARGET='tests/test_bmr_vs_vl_refit.py' \
#          cluster/sbatch/pytest_sharded.sbatch
#
# The array size IS the shard count -- it is read from SLURM_ARRAY_TASK_COUNT
# (override with PYTEST_SHARDS if your Slurm does not export it).
#
# Outputs:
#   cluster/logs/pytest_shard_<arrayjob>_<task>.{out,err}
#
# Aggregating: every shard prints a "SHARD RESULT" line. Collect with
#   grep -h 'SHARD RESULT' cluster/logs/pytest_shard_<arrayjob>_*.out
# and check the coverage gate -- the assigned counts MUST sum to the collected
# total, or a shard silently ran nothing.
#
# NEVER pip install inside this job (shared venv; concurrent resolvers corrupt
# it). Provision once from the login node -- see cluster/README.md.
# =============================================================================

#SBATCH --job-name=pytest_shard
#SBATCH --output=cluster/logs/pytest_shard_%A_%a.out
#SBATCH --error=cluster/logs/pytest_shard_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

source cluster/lib/cluster_env.sh
crlf_guard

setup_torch_threads 4
activate_env
verify_torch
print_job_header "Sharded pytest"

TEST_TARGET="${TEST_TARGET:-tests/}"
# PYTEST_MARK is kept separate from TEST_TARGET: TEST_TARGET is word-split (it
# is a list of paths), so a marker expression inside it gets shredded and
# pytest exits 4 with "no tests ran".
PYTEST_MARK="${PYTEST_MARK:-}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${PYTEST_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

echo "TEST_TARGET=${TEST_TARGET}"
echo "PYTEST_MARK=${PYTEST_MARK:-<none>}"
echo "SHARD=${SHARD_ID} of ${NSHARDS}"

mkdir -p cluster/logs

MARK_ARGS=()
if [[ -n "$PYTEST_MARK" ]]; then
    MARK_ARGS=(-m "$PYTEST_MARK")
fi

# --- Collect the deterministic test list -------------------------------------
# `--collect-only -q` prints one node id per line plus a trailing summary; keep
# only lines containing "::". pytest's collection order is deterministic for a
# fixed tree, so every shard derives the identical list without coordination.
COLLECT_FILE="$(mktemp)"
if ! python -m pytest ${TEST_TARGET} "${MARK_ARGS[@]}" \
        --collect-only -q -p no:cacheprovider 2>/dev/null \
        | grep '::' > "$COLLECT_FILE"; then
    echo "ERROR: collection produced no tests for TEST_TARGET='${TEST_TARGET}' PYTEST_MARK='${PYTEST_MARK}'"
    rm -f "$COLLECT_FILE"
    exit 1
fi

TOTAL=$(wc -l < "$COLLECT_FILE")
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: 0 tests collected -- check the marker expression."
    rm -f "$COLLECT_FILE"
    exit 1
fi

# --- Stripe: this shard takes every NSHARDS-th test starting at SHARD_ID ------
SHARD_FILE="$(mktemp)"
awk -v s="$SHARD_ID" -v n="$NSHARDS" 'NR % n == s' "$COLLECT_FILE" > "$SHARD_FILE"
ASSIGNED=$(wc -l < "$SHARD_FILE")

echo "Collected ${TOTAL} tests; this shard runs ${ASSIGNED}."
if [[ "$ASSIGNED" -eq 0 ]]; then
    # Legitimate when shards > tests. Announce loudly rather than exiting 0
    # silently, so the coverage gate can tell this apart from a lost shard.
    echo "SHARD RESULT shard=${SHARD_ID}/${NSHARDS} assigned=0 exit=0 (no tests assigned)"
    rm -f "$COLLECT_FILE" "$SHARD_FILE"
    exit 0
fi

mapfile -t SHARD_TESTS < "$SHARD_FILE"
rm -f "$COLLECT_FILE" "$SHARD_FILE"

python -m pytest "${SHARD_TESTS[@]}" -q -p no:cacheprovider
EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════"
echo "SHARD RESULT shard=${SHARD_ID}/${NSHARDS} assigned=${ASSIGNED} total=${TOTAL} exit=${EXIT_CODE}"
echo "════════════════════════════════════════════════════════"
exit $EXIT_CODE
