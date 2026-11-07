# Running Forward Tracing & Paper Analysis on NERSC Perlmutter

This folder contains the Julia runners and Slurm job scripts for running incremental production-scale $\mathrm{O_2^+}$ forward-tracing ensembles and paper analyses (`Figures 3–7`) on NERSC Perlmutter.

## Files in This Folder

- [`setup_perlmutter.sh`](setup_perlmutter.sh): One-time setup script on a Perlmutter login node (`Pkg.instantiate()`, `Pkg.precompile()`, and converting `BATSRUS-data/` `.dat` files into `data/mars_fields_with_crustal.vts` and `data/mars_fields_without_crustal.vts`).
- [`run_perlmutter_chunk.jl`](run_perlmutter_chunk.jl): Runs one chunk (`SLURM_ARRAY_TASK_ID`) or a sequential range of chunks (`CHUNK_START..CHUNK_END`) with hardware-entropy `RandomDevice()` seeds, SHA-256 256-bit `Xoshiro` streams, non-overlapping global `particle_id`s, and compact `Float32` packed JLD2 trajectory output (`format_version = 3`).
- [`analyze_perlmutter_chunks.jl`](analyze_perlmutter_chunks.jl): Discovers all completed `*_chunk_XXX` directories, verifies seed/chunk uniqueness, pools them with automatic $1/K$ rate-weight scaling, generates `paper_analysis.jld2` (`~3.2 MB`) and all 6 paper figures (`Figures 3–7`), and computes chunk-by-chunk statistical convergence (`*_convergence.toml` and `*_convergence.png`).
- [`submit_mc_cpu.slurm`](submit_mc_cpu.slurm): Slurm batch script for Perlmutter CPU nodes (`--constraint=cpu`, 128 threads).
- [`submit_mc_gpu.slurm`](submit_mc_gpu.slurm): Slurm batch script for Perlmutter GPU nodes (`--constraint=gpu`, A100 GPU with `TRACING_BACKEND=cuda`).
- [`submit_analyze.slurm`](submit_analyze.slurm): Slurm batch script for pooling completed chunks and generating the paper figures and convergence report.

---

## Step-by-Step Workflow on Perlmutter

### 1. One-Time Setup (Login Node)

```bash
cd /path/to/MarsTP.jl
bash examples/forward_tracing/perlmutter/setup_perlmutter.sh
```

### 2. Submit Incremental Chunks (Slurm Job Array or Single Job)

All outputs default to `$SCRATCH/MarsTP_outputs/mc_400km_${CASE}_chunk_XXX` so large trajectory files stay on high-bandwidth Lustre scratch rather than `$HOME`.

- **Round 1 (`1` chunk of `1,015,300` particles, `per_cell = 100`, `~1.6 GB`)**:
  ```bash
  sbatch --account=<YOUR_ACCT> --array=1-1 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
  ```
- **Add Chunks `2..5` in Parallel (`5,076,500` cumulative particles, `~8 GB` total)**:
  ```bash
  sbatch --account=<YOUR_ACCT> --array=2-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
  ```
- **Full `101.53M`-Particle Run (`10,000` particles/cell across `10` parallel chunks of `per_cell = 1000`)**:
  ```bash
  PER_CELL=1000 sbatch --account=<YOUR_ACCT> --array=1-10 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
  ```
- **Run Companion Without-Crustal-Field Case**:
  ```bash
  CASE=without_crustal sbatch --account=<YOUR_ACCT> --array=1-5 examples/forward_tracing/perlmutter/submit_mc_cpu.slurm
  ```

### 3. Pool Chunks, Check Convergence, & Generate Paper Figures

You can either chain the analysis job automatically using `--dependency=afterok:$JOBID` or submit it after any round finishes:

```bash
sbatch --account=<YOUR_ACCT> examples/forward_tracing/perlmutter/submit_analyze.slurm
```

This creates:
- `$SCRATCH/MarsTP_outputs/mc_400km_with_crustal_cum_001_paper/` .. `_cum_00K_paper/` (each only `~3.2 MB`, containing `paper_analysis.jld2` and all 6 PNG figures)
- `$SCRATCH/MarsTP_outputs/mc_400km_with_crustal_convergence.toml`
- `$SCRATCH/MarsTP_outputs/mc_400km_with_crustal_convergence.png`

### 4. Copy Compact Analysis Results (~3.2 MB) Back to Local Laptop

Because `paper_analysis.jld2` + the 6 PNG figures + `*_convergence.*` total `< 4 MB`, you can copy the entire reduced dataset to your laptop in seconds without transferring the raw `.jld2` trajectory batches:

```bash
scp -r perlmutter:$SCRATCH/MarsTP_outputs/mc_400km_*_paper perlmutter:$SCRATCH/MarsTP_outputs/*_convergence.* ./outputs/
```
