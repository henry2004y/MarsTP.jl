#!/bin/bash
# Fetch Compact Paper Analysis Results from NERSC Perlmutter (< 4 MB total)
# Usage: bash examples/forward_tracing/perlmutter/fetch_results.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE_HOST="perlmutter"
LOCAL_OUT="${REPO_ROOT}/outputs/perlmutter_runs"

mkdir -p "${LOCAL_OUT}"

echo "=== 1. Resolving remote SCRATCH directory ==="
REMOTE_SCRATCH=$(ssh "${REMOTE_HOST}" "printenv SCRATCH")
if [ -z "${REMOTE_SCRATCH}" ]; then
    echo "ERROR: Could not query SCRATCH on ${REMOTE_HOST}."
    exit 1
fi

echo "=== 2. Fetching Compact Paper Analysis & Figures from Perlmutter ==="
rsync -avz --progress \
    --include="*/" \
    --include="*paper/**" \
    --include="*_convergence.*" \
    --exclude="*" \
    "${REMOTE_HOST}:${REMOTE_SCRATCH}/MarsTP_outputs/" "${LOCAL_OUT}/"

echo "=== Results successfully downloaded to ${LOCAL_OUT} ==="
ls -lh "${LOCAL_OUT}"
