#!/bin/bash
#SBATCH --job-name=marstp_pre_cpu
#SBATCH --account=m4453
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=01:30:00
#SBATCH --output=logs/pre_cpu_%A_%a.out
#SBATCH --error=logs/pre_cpu_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

command -v julia &>/dev/null || module load julia 2>/dev/null || true

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-128}"
export MPLBACKEND="Agg"

CASE="${CASE:-with_crustal}"
PER_CELL="${PER_CELL:-500}"
CELL_STRIDE="${CELL_STRIDE:-1}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:-2}"
OUT_PREFIX="${OUT_PREFIX:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_${CASE}}"

if [[ "$CASE" == "without_crustal" ]]; then
    FIELDS_PATH="data/mars_fields_without_crustal.vts"
else
    FIELDS_PATH="data/mars_fields_with_crustal.vts"
fi

julia --threads="$JULIA_NUM_THREADS" --project=. \
    examples/forward_tracing/perlmutter/precompute_chunk_worker.jl \
    "$OUT_PREFIX" "$CHUNK_ID" "$PER_CELL" "$CELL_STRIDE" "$FIELDS_PATH"

