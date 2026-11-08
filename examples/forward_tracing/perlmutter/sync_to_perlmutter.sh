#!/bin/bash
# Sync MarsTP.jl and precomputed VTS grids from local WSL to NERSC Perlmutter ($SCRATCH/MarsTP.jl)
# Usage: bash examples/forward_tracing/perlmutter/sync_to_perlmutter.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_HOST="perlmutter"

echo "=== 1. Checking SSH connection to ${REMOTE_HOST} ==="
ssh -O check "${REMOTE_HOST}" 2>/dev/null || {
    echo "ControlMaster socket not active. Establishing connection..."
    echo "Please authenticate with your password + MFA if prompted."
    ssh -fN "${REMOTE_HOST}"
}

echo "=== 2. Resolving remote SCRATCH directory ==="
REMOTE_SCRATCH=$(ssh "${REMOTE_HOST}" "printenv SCRATCH")
if [ -z "${REMOTE_SCRATCH}" ]; then
    echo "ERROR: Could not query SCRATCH on ${REMOTE_HOST}."
    exit 1
fi
REMOTE_BASE="${REMOTE_SCRATCH}/MarsTP.jl"
echo "Target remote path: ${REMOTE_BASE}"

echo "=== 3. Creating remote directory structure on Perlmutter ==="
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_BASE}/data' '${REMOTE_SCRATCH}/MarsTP_outputs'"

echo "=== 4. Syncing repository code to Perlmutter ==="
rsync -avz --progress \
    --exclude '.git' \
    --exclude 'outputs' \
    --exclude '*.jld2' \
    --exclude 'BATSRUS-data' \
    --exclude 'data/*.vts' \
    "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_BASE}/"

echo "=== 5. Syncing precomputed .vts grids and .mat source rate files to data/ ==="
rsync -avz --progress \
    "${REPO_ROOT}/data/"*.vts \
    "${REPO_ROOT}/data/"*.mat \
    "${REMOTE_HOST}:${REMOTE_BASE}/data/"

echo "=== 6. Running Julia environment setup on Perlmutter ==="
ssh "${REMOTE_HOST}" "bash -lc 'cd ${REMOTE_BASE} && bash examples/forward_tracing/perlmutter/setup_perlmutter.sh'"

echo "=== Sync and setup complete! ==="
