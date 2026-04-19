#!/bin/bash
# =============================================================================
# SLURM Job: Auto-Push Phase 16 Acceptance Results to Git
# =============================================================================
# Creates a results branch, commits Phase 16 cluster logs + result summaries,
# and pushes to the remote. Submit as a final dependent job after
# cluster/run_phase16_acceptance.slurm (use `afterany`, not `afterok`, so the
# push happens even if the acceptance test fails -- we still want the logs).
#
# Strategy (default -- branch push):
#   1. Creates branch: results/phase16-acceptance-YYYYMMDD-HHMMSS
#   2. Stages cluster/logs/phase16_* and cluster/results/phase16_*
#   3. Commits with descriptive message including parent SLURM job IDs
#   4. Pushes branch to GIT_REMOTE (default: origin)
#   5. Returns to the original branch
#   User merges when satisfied:
#     git fetch origin && git merge origin/results/phase16-acceptance-...
#
# Strategy (opt-in -- PUSH_TO_MAIN=true):
#   Runs git pull --rebase before committing. Riskier: merge conflicts fail
#   silently and leave results un-pushed. Not recommended.
#
# Usage:
#   # Dispatched automatically by cluster/submit_phase16.sh via --parsable
#   # dependency chaining:
#   bash cluster/submit_phase16.sh
#
#   # Manual submission after a specific job:
#   sbatch --dependency=afterany:12345 \
#       --export=ALL,PARENT_JOBS=12345 \
#       cluster/99_push_phase16_results.slurm
#
#   # Override branch name:
#   sbatch --export=BRANCH_NAME=results/phase16-rerun-2026-04-19 \
#       cluster/99_push_phase16_results.slurm
#
#   # Push directly to main (risky, not recommended):
#   sbatch --export=PUSH_TO_MAIN=true cluster/99_push_phase16_results.slurm
#
#   # Override remote (e.g. if origin is not your fork):
#   sbatch --export=GIT_REMOTE=upstream cluster/99_push_phase16_results.slurm
#
# Prerequisites:
#   - SSH key / deploy key configured for git push on compute nodes.
#     Recommended: ~/.ssh/config Host alias that uses your deploy key for
#     github.com (see cluster/README.md Form B) so git push works without
#     GIT_SSH_COMMAND env vars.
#   - Git remote configured (defaults to 'origin').
#
# WINDOWS / CRLF GOTCHA:
#   Git on Windows can convert LF -> CRLF on checkout. sbatch rejects scripts
#   with CRLF line endings. Fix this in the submit wrapper BEFORE any sbatch
#   call:
#     sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true
#   cluster/submit_phase16.sh does this automatically.
#
# Derived from: project_utils/templates/hpc/slurm_push_results_TEMPLATE.slurm
# =============================================================================

#SBATCH --job-name=push_phase16_results
#SBATCH --output=cluster/logs/push_phase16_%j.out
#SBATCH --error=cluster/logs/push_phase16_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

# =============================================================================
# Configuration
# =============================================================================
PUSH_TO_MAIN="${PUSH_TO_MAIN:-false}"
BRANCH_NAME="${BRANCH_NAME:-results/phase16-acceptance-$(date +%Y%m%d-%H%M%S)}"
PARENT_JOBS="${PARENT_JOBS:-${SLURM_JOB_DEPENDENCY:-unknown}}"
# Override if 'origin' is not your personal fork (e.g. GIT_REMOTE=upstream).
GIT_REMOTE="${GIT_REMOTE:-origin}"

# =============================================================================
# Environment Setup (minimal -- only need git)
# =============================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs

echo "============================================================"
echo "Auto-Push Phase 16 Acceptance Results"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:     $PROJECT_ROOT"
echo "Remote:      $GIT_REMOTE"
echo "Strategy:    $(if [[ "$PUSH_TO_MAIN" == "true" ]]; then echo 'push to main'; else echo "branch: $BRANCH_NAME"; fi)"
echo "Parent jobs: $PARENT_JOBS"
echo "Start:       $(date)"
echo "============================================================"

# =============================================================================
# Verify Git Setup
# =============================================================================
echo ""
echo "Checking git configuration..."

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found on this node"
    exit 1
fi

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not inside a git repository: $PROJECT_ROOT"
    exit 1
fi

if ! git remote get-url "$GIT_REMOTE" &>/dev/null; then
    echo "ERROR: No '$GIT_REMOTE' remote configured"
    echo "Available remotes: $(git remote)"
    echo "Override with: sbatch --export=GIT_REMOTE=<name> ..."
    exit 1
fi

echo "Testing push access to $GIT_REMOTE..."
if ! git ls-remote --exit-code "$GIT_REMOTE" HEAD &>/dev/null 2>&1; then
    echo "ERROR: Cannot reach $GIT_REMOTE remote"
    echo "Check SSH keys or credential helper on cluster nodes."
    echo "Remote URL: $(git remote get-url "$GIT_REMOTE")"
    exit 1
fi
echo "  Remote accessible: $(git remote get-url "$GIT_REMOTE")"

CURRENT_BRANCH=$(git branch --show-current)
echo "  Current branch: $CURRENT_BRANCH"
echo "  HEAD:           $(git rev-parse --short HEAD)"
echo ""

# =============================================================================
# Stage Phase 16 Result Files
# =============================================================================
echo "Staging Phase 16 result files..."

STAGED_COUNT=0

stage_files() {
    local pattern="$1"
    local label="$2"
    local count
    count=$(git add --dry-run -- $pattern 2>/dev/null | wc -l)
    if [[ "$count" -gt 0 ]]; then
        git add -- $pattern 2>/dev/null
        echo "  + $label ($count files)"
        STAGED_COUNT=$((STAGED_COUNT + count))
    fi
}

# Phase 16 cluster artifacts only. NEVER stage figures/, benchmarks/fixtures/,
# or the bilinear benchmark JSON -- those paths are gitignored.
stage_files "cluster/logs/phase16_*.out"             "Phase 16 SLURM stdout logs"
stage_files "cluster/logs/phase16_*.err"             "Phase 16 SLURM stderr logs"
stage_files "cluster/logs/pytest_phase16_*.log"      "Phase 16 pytest tee logs"
stage_files "cluster/results/phase16_acceptance_*.json"  "Phase 16 acceptance result summaries"
stage_files "cluster/results/phase16_acceptance_*.md"    "Phase 16 acceptance markdown reports"

echo ""

# Check if there's anything to commit.
if git diff --cached --quiet; then
    echo "No new or modified Phase 16 result files to commit."
    echo "STATUS: NOTHING TO PUSH"
    exit 0
fi

echo "Staged changes:"
git diff --cached --stat
echo ""

# =============================================================================
# Commit and Push
# =============================================================================
if [[ "$PUSH_TO_MAIN" == "true" ]]; then
    echo "Pulling latest main with rebase..."
    if ! git pull --rebase "$GIT_REMOTE" main; then
        echo "ERROR: git pull --rebase failed (likely merge conflict)"
        echo "Results are staged but NOT pushed. Manual resolution required."
        exit 1
    fi

    git commit -m "results(phase16): acceptance gate outputs (SLURM jobs: ${PARENT_JOBS})"

    echo "Pushing to main on $GIT_REMOTE..."
    if git push "$GIT_REMOTE" main; then
        echo "STATUS: PUSHED TO MAIN"
    else
        echo "ERROR: git push to main failed"
        exit 1
    fi
else
    echo "Creating results branch: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"

    git commit -m "results(phase16): acceptance gate outputs (SLURM jobs: ${PARENT_JOBS})

Branch: $BRANCH_NAME
Node: ${SLURMD_NODENAME:-$(hostname)}
Date: $(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)

Merge with:
  git checkout main && git merge $BRANCH_NAME"

    echo "Pushing branch to $GIT_REMOTE..."
    if git push -u "$GIT_REMOTE" "$BRANCH_NAME"; then
        echo "STATUS: PUSHED TO BRANCH $BRANCH_NAME"
    else
        echo "ERROR: git push failed"
        exit 1
    fi

    echo "Returning to branch: $CURRENT_BRANCH"
    git checkout "$CURRENT_BRANCH"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Git Push Complete!"
echo "============================================================"
echo "End time: $(date)"
if [[ "$PUSH_TO_MAIN" == "true" ]]; then
    echo "Pull locally: git pull $GIT_REMOTE main"
else
    echo "Merge locally: git fetch $GIT_REMOTE && git merge $GIT_REMOTE/$BRANCH_NAME"
fi
echo "============================================================"
