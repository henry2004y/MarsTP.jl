#!/bin/bash
# Launch Production Scale Runs on NERSC Perlmutter
# Usage: bash examples/forward_tracing/perlmutter/launch_production.sh [benchmark|production]

set -euo pipefail

MODE="${1:-benchmark}"
ACCOUNT="m4453"

echo "=== MarsTP.jl Perlmutter Production Launch (${MODE}) ==="
echo "Account: ${ACCOUNT}"

if [ "${MODE}" = "benchmark" ]; then
    echo "--- 1. Submitting CPU Pilot Chunk (1 chunk, ~1.015M particles) ---"
    export OUT_PREFIX="${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_benchmark_cpu"
    CPU_JOB=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,OUT_PREFIX="${OUT_PREFIX}" --qos=debug --time=00:30:00 --array=1-1 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm)
    echo "Submitted CPU Job: ${CPU_JOB} (Output: ${OUT_PREFIX}_chunk_001)"

    echo "--- 2. Submitting GPU Test Chunk (1 chunk, ~1.015M particles on A100) ---"
    export OUT_PREFIX="${SCRATCH:-$PWD/outputs}/MarsTP_outputs/mc_400km_benchmark_gpu"
    GPU_JOB=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,OUT_PREFIX="${OUT_PREFIX}" --qos=debug --time=00:30:00 --array=1-1 examples/forward_tracing/perlmutter/submit_mc_gpu.slurm)
    echo "Submitted GPU Job: ${GPU_JOB} (Output: ${OUT_PREFIX}_chunk_001)"

    echo ""
    echo "Monitor jobs via: squeue -u \$USER"
    echo "Compare log outputs in logs/ once completed."

elif [ "${MODE}" = "production" ]; then
    echo "--- 1. Submitting 5-Chunk Array: with_crustal (~5.08M particles) ---"
    JOB_CRUSTAL=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,CASE=with_crustal --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm)
    echo "Submitted with_crustal Job Array: ${JOB_CRUSTAL}"

    echo "--- 2. Chaining Analysis Job for with_crustal ---"
    ANALYSIS_CRUSTAL=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,CASE=with_crustal --dependency=afterok:${JOB_CRUSTAL} examples/forward_tracing/perlmutter/submit_analyze.slurm)
    echo "Chained with_crustal Analysis Job: ${ANALYSIS_CRUSTAL}"

    echo "--- 3. Submitting 5-Chunk Array: without_crustal (~5.08M particles) ---"
    JOB_NO_CRUSTAL=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,CASE=without_crustal --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm)
    echo "Submitted without_crustal Job Array: ${JOB_NO_CRUSTAL}"

    echo "--- 4. Chaining Analysis Job for without_crustal ---"
    ANALYSIS_NO_CRUSTAL=$(sbatch --parsable --account="${ACCOUNT}" --export=ALL,CASE=without_crustal --dependency=afterok:${JOB_NO_CRUSTAL} examples/forward_tracing/perlmutter/submit_analyze.slurm)
    echo "Chained without_crustal Analysis Job: ${ANALYSIS_NO_CRUSTAL}"

    echo ""
    echo "All production arrays and analysis jobs submitted!"
    echo "Monitor progress: squeue -u \$USER"
fi

