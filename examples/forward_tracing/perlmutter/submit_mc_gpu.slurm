#!/bin/bash
#SBATCH --job-name=marstp_mc_gpu
#SBATCH --account=m4453
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/mc_gpu_%A_%a.out
#SBATCH --error=logs/mc_gpu_%A_%a.err

# Perlmutter GPU Forward Tracing Job Script (A100 GPU)
#
# Example:
#   sbatch --account=<YOUR_ACCT> --array=1-5 examples/forward_tracing/perlmutter/submit_mc_gpu.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

command -v julia &>/dev/null || module load julia 2>/dev/null || true

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export CASE="${CASE:-with_crustal}"
export PER_CELL="${PER_CELL:-100}"
export CELL_STRIDE="${CELL_STRIDE:-1}"
export ALTITUDE_KM="${ALTITUDE_KM:-400.0}"
export DT_S="${DT_S:-0.1}"
export TMAX_S="${TMAX_S:-1000.0}"
export BATCH_SIZE="${BATCH_SIZE:-16384}"
export SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-roi_and_sample}"
export SAVE_STRIDE="${SAVE_STRIDE:-5}"
export SKIP_INNER_TRAPPED="${SKIP_INNER_TRAPPED:-true}"
export TRACING_BACKEND="cuda"
export RANDOM_SEED="${RANDOM_SEED:-true}"
export OUT_PREFIX="${OUT_PREFIX:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_${CASE}}"
export MC_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"

julia --threads="$JULIA_NUM_THREADS" --project=. examples/forward_tracing/perlmutter/run_perlmutter_chunk.jl
