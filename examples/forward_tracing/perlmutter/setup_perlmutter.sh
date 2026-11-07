#!/bin/bash
# One-time setup script on a Perlmutter login node.
# Usage: bash examples/forward_tracing/perlmutter/setup_perlmutter.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs "${SCRATCH:-$REPO_ROOT/outputs}/MarsTP_outputs"

echo "=== 1. Instantiating and precompiling MarsTP.jl environment ==="
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

if [[ ! -f "data/mars_fields_with_crustal.vts" || ! -f "data/mars_fields_without_crustal.vts" ]]; then
    echo "=== 2. Converting BATSRUS .dat files to spherical .vts in data/ ==="
    julia --project=. scripts/convert_batsrus_mhd.jl
else
    echo "=== 2. Found existing data/mars_fields_with_crustal.vts and data/mars_fields_without_crustal.vts ==="
fi

echo "=== Perlmutter setup complete! Ready to submit jobs from $REPO_ROOT ==="
