#!/bin/bash
#SBATCH --job-name=marstp_analyze
#SBATCH --account=m4453
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err

# Perlmutter Multi-Chunk Pooling, Paper Figures (Figs. 3-7), & Convergence Analysis
#
# Can be chained automatically after a tracing job array finishes:
#   JOBID=$(sbatch --parsable --account=<YOUR_ACCT> --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm)
#   sbatch --account=<YOUR_ACCT> --dependency=afterok:$JOBID examples/forward_tracing/perlmutter/submit_analyze.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

command -v julia &>/dev/null || module load julia 2>/dev/null || true

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-64}"
export CASE="${CASE:-with_crustal}"
export ANALYZE_MODE="${ANALYZE_MODE:-incremental}"
export OUT_PREFIX="${OUT_PREFIX:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_${CASE}}"
export MPLBACKEND="Agg"

julia --threads="$JULIA_NUM_THREADS" --project=. examples/forward_tracing/perlmutter/analyze_perlmutter_chunks.jl "$OUT_PREFIX" "$ANALYZE_MODE"
