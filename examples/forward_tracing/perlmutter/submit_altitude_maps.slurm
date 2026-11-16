#!/bin/bash
#SBATCH --job-name=marstp_alt_maps
#SBATCH --account=m4453
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --time=00:45:00
#SBATCH --output=logs/alt_maps_%j.out
#SBATCH --error=logs/alt_maps_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

command -v julia &>/dev/null || module load julia 2>/dev/null || true

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-128}"
export MPLBACKEND="Agg"

CASE="${CASE:-without_crustal}"
FLUX_MODEL="${FLUX_MODEL:-surface_flux_maxwellian}"
PER_CELL="${PER_CELL:-400}"
N_CHUNKS="${N_CHUNKS:-2}"
RECOMPUTE="${RECOMPUTE:-true}"
DEFAULT_FIELDS="data/mars_fields_${CASE}.vts"
FIELDS_PATH="${FIELDS_PATH:-$DEFAULT_FIELDS}"
OUT_DIR="${OUT_DIR:-${SCRATCH:-$PWD/outputs}/MarsTP_outputs/surface_flux_400km/${CASE}_altitude_maps}"
OUT_PNG="${OUT_DIR}/altitude_energy_maps_025_050_075_100Rm.png"

mkdir -p "$OUT_DIR"

echo "=== Altitude energy maps: CASE=$CASE, FLUX_MODEL=$FLUX_MODEL, PER_CELL=$PER_CELL, N_CHUNKS=$N_CHUNKS, OUT_DIR=$OUT_DIR ==="
julia --threads="$JULIA_NUM_THREADS" --project=. \
    examples/forward_tracing/monte_carlo_forward_tracing/plot_altitude_energy_maps.jl \
    "$OUT_PNG" "$FIELDS_PATH" "$PER_CELL" "$N_CHUNKS" "$RECOMPUTE" "$FLUX_MODEL"
