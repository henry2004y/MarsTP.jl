#!/bin/bash
#SBATCH --job-name=marstp_inc_cpu
#SBATCH --account=m4453
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --output=logs/inc_cpu_%j.out
#SBATCH --error=logs/inc_cpu_%j.err

# Perlmutter Incremental Multi-Chunk Forward Tracing + Auto-Convergence Runner
#
# Runs chunks sequentially on a 128-core (256-thread) CPU node, distilling each
# chunk into its compact `paper_analysis.jld2` summary, deleting raw trajectory
# batches for chunks >= 2 (keeping chunk 1 raw batches), merging cumulative chunks
# 1..k in O(1) time via `merge_paper_analyses`, and stopping automatically once
# all key observables converge within `REL_TOL` (default: 0.03 = 3%).
#
# Usage:
#   CASE=with_crustal sbatch examples/forward_tracing/perlmutter/submit_incremental_cpu.slurm
#   CASE=without_crustal sbatch examples/forward_tracing/perlmutter/submit_incremental_cpu.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

command -v julia &>/dev/null || module load julia 2>/dev/null || true

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-128}"
export MPLBACKEND="Agg"

CASE="${CASE:-with_crustal}"
N_CHUNKS="${N_CHUNKS:-20}"
PER_CELL="${PER_CELL:-500}"
CELL_STRIDE="${CELL_STRIDE:-1}"
SAVE_MODE="${SAVE_MODE:-roi_and_sample}"
REL_TOL="${REL_TOL:-0.03}"
OUT_PREFIX="${OUT_PREFIX:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_${CASE}}"

if [[ "$CASE" == "without_crustal" ]]; then
    FIELDS_PATH="data/mars_fields_without_crustal.vts"
else
    FIELDS_PATH="data/mars_fields_with_crustal.vts"
fi

echo "=================================================================="
echo "Starting Incremental Auto-Converging Run on Perlmutter"
echo "  CASE        = $CASE"
echo "  OUT_PREFIX  = $OUT_PREFIX"
echo "  FIELDS_PATH = $FIELDS_PATH"
echo "  N_CHUNKS    = $N_CHUNKS (max)"
echo "  PER_CELL    = $PER_CELL"
echo "  REL_TOL     = $REL_TOL"
echo "  THREADS     = $JULIA_NUM_THREADS"
echo "=================================================================="

julia --threads="$JULIA_NUM_THREADS" --project=. \
    examples/forward_tracing/monte_carlo_forward_tracing/run_incremental_chunks.jl \
    "$OUT_PREFIX" "$N_CHUNKS" "$PER_CELL" "$CELL_STRIDE" "$SAVE_MODE" "$REL_TOL" "$FIELDS_PATH"
