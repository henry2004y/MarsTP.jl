#!/bin/bash
#SBATCH --job-name=marstp_mc_cpu
#SBATCH --account=mXXXX
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --output=logs/mc_cpu_%A_%a.out
#SBATCH --error=logs/mc_cpu_%A_%a.err

# Perlmutter CPU Forward Tracing Job Script
#
# Examples:
#   1) Run 1 chunk of 1M particles (per_cell=100):
#      sbatch --account=<YOUR_ACCT> examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
#
#   2) Run 5 incremental chunks of 1M particles in parallel (5M total) via Slurm job array:
#      sbatch --account=<YOUR_ACCT> --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
#
#   3) Run 10 chunks of 10M particles in parallel (101.53M total = 10,000/cell):
#      PER_CELL=1000 sbatch --account=<YOUR_ACCT> --array=1-10 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
#
#   4) Run the without-crustal-field case:
#      CASE=without_crustal sbatch --account=<YOUR_ACCT> --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-128}"
export CASE="${CASE:-with_crustal}"
export PER_CELL="${PER_CELL:-100}"
export CELL_STRIDE="${CELL_STRIDE:-1}"
export ALTITUDE_KM="${ALTITUDE_KM:-400.0}"
export DT_S="${DT_S:-0.1}"
export TMAX_S="${TMAX_S:-1000.0}"
export SAVE_TRAJECTORIES="${SAVE_TRAJECTORIES:-roi_and_sample}"
export SAVE_STRIDE="${SAVE_STRIDE:-5}"
export SKIP_INNER_TRAPPED="${SKIP_INNER_TRAPPED:-true}"
export TRACING_BACKEND="cpu"
export RANDOM_SEED="${RANDOM_SEED:-true}"
export OUT_PREFIX="${OUT_PREFIX:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_${CASE}}"
export MC_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unavailable)"

julia --threads="$JULIA_NUM_THREADS" --project=. examples/forward_tracing/perlmutter/run_perlmutter_chunk.jl
