# MarsTP.jl

MarsTP.jl is a Mars test-particle modeling package built on [TestParticle.jl](https://github.com/henry2004y/TestParticle.jl) (v0.26), [BorisPushers.jl](https://github.com/henry2004y/BorisPushers.jl) (v0.2), and [Batsrus.jl](https://github.com/henry2004y/Batsrus.jl). It combines BATSRUS 3D MHD electromagnetic fields and plasma moments with GITM and AMPS atmospheric inputs to support forward and backward particle tracing, velocity-distribution analysis at fixed locations, multi-channel electric-field and guiding-center work calculations, and ion escape analysis. All visualization and plotting utilities are written in Julia using `PyPlot`.

## Reproducible figure examples

See [examples/README.md](examples/README.md) for the 800 km O2+ trajectory panels
and the dayside/nightside electric-work maps, including the Julia source and plotting scripts,
the PNG figures, prerequisites, run commands, and interpretation notes.

Saved forward trajectories can be analyzed with `forward_psd_saved`, using the
same accumulator as `forward_psd`. `write_trajectory_batch` provides bounded
batch output for large ensembles. See the [Monte Carlo API and reproducible
three-probe example](examples/forward_tracing/monte_carlo_forward_tracing/README.md).

## Data and BATSRUS MHD Conversion

Required runtime inputs live in `data/`:

- `amps_sph.mat` (AMPS hot atomic oxygen grid)
- `gitm_sph.mat` (GITM neutral atmosphere grid)
- `O2plus_source_rates.mat` (plus converted `O2plus_source_rates_with_crustal.mat` and `O2plus_source_rates_without_crustal.mat`)

Raw 3D BATSRUS Tecplot outputs live in `BATSRUS-data/`:

- `BATSRUS-data/MHD-with-crustal/3d__var_1_n00150000.dat`
- `BATSRUS-data/MHD-without-crustal/3d__var_1_n00120000.dat`

The converted `.vts` files are not tracked in Git due to size. Run `scripts/convert_batsrus_mhd.jl` to convert both BATSRUS `.dat` files into structured spherical `.vts` files (`120 × 72 × 144`, from 200 km altitude to $4\,R_M$) and generate their corresponding `.mat` source-rate files:

```sh
julia --project=. scripts/convert_batsrus_mhd.jl
```

This produces:

- `data/mars_fields_with_crustal.vts` (and copies to default `data/mars_fields_spherical_from_dat.vts`)
- `data/mars_fields_without_crustal.vts`
- `data/O2plus_source_rates_with_crustal.mat`
- `data/O2plus_source_rates_without_crustal.mat`

Each converted `.vts` file stores reconstructed Ohm's-law electric field components (`E_Total [V/m]`, `E_conv [V/m]`, `E_hall [V/m]`, `E_gradpe [V/m]`, `E_res [V/m]`, and diagnostic `E_plot [V/m]`) along with `B_Field [T]` and plasma moments (`Hplus = Hswp + Hpp`, `Oplus`, `O2plus`, `CO2plus`, `T_e`, `n_e`). You can also pass a `.dat` path directly to `load_mhd_fields` or `load_mhd_moments`, or interpolate 2D slices via `interpolate_batsrus_slice`.

## Basic Use

Fixed-step forward tracing can use NVIDIA GPUs through `TestParticle` v0.26 and `BorisPushers` v0.2:
see [CPU/GPU example, benchmark and limitations](examples/gpu/README.md).
Pass `backend=CUDA.CUDABackend()` and `solver=:boris` to `trace_forward`.
Use `trace_forward_bounded` for independent particle stopping at the 200 km
absorption and outer escape spheres, with termination states and times.
It can accumulate total, convection and Hall electric work on the GPU with
`work_itp`. The production shell driver supports `tracing_backend=:cuda` and
direct device-summary output with `work_mode=:summary`.
The source-accumulating backward VDF remains on CPU.

Monte Carlo Maxwellian initial states and physical weights are available via
`sample_maxwellian_source`. See [weight definitions and units](examples/forward_tracing/monte_carlo_weights.md)
and the [local MHD source example](examples/forward_tracing/maxwellian_source.jl).
For a finite-volume detector, see the [3D phase-space density derivation](examples/forward_tracing/monte_carlo_forward_tracing/README.md),
including residence-time weighting, units, and the distinction from source density weights.
Use `forward_psd(solutions; detector_m, side_m, vlim, vgrid, species,
rate_weights_s, option="3D")` to compute the detector PSD from saved forward
trajectories. `vgrid` is the bin count per axis; `vlim` is in m/s by default.
Options `"Vx-Vy"`, `"Vy-Vz"`, and `"Vx-Vz"` return integrated 2D distributions.
See the guide above for units, source-rate normalization, and a complete example.

```julia
using MarsTP

cfg = BacktraceConfig(
    detector_Rm = SA[-1.5, 0.0, 1.0],
    vx_min_kms = -500,
    vx_max_kms = 500,
    vy_min_kms = -500,
    vy_max_kms = 500,
    vz_min_kms = -500,
    vz_max_kms = 500,
    dv_kms = 1,
    dvy_kms = 1,
    solver = :boris,
    dt = -0.2,
    stream_vy = true,
)

result = run_backtrace_vdf(cfg)
moments = vdf_moments(result)
refined_cfg = refine_backtrace_config(result; dv_kms = 0.5, dvy_kms = 0.5)
plot_vx_vz(result; output = "outputs/vdf_xz.png")
```

The returned `f2d_xz` is the 3D VDF integrated over `Vy` using `sum(f) * dvy`.
`vdf_moments` wraps `TestParticle.velocity_moments` to compute density `n_m3` and bulk velocity components `(Vx_kms, Vz_kms)`, and `refine_backtrace_config` wraps `TestParticle.refine_vdf_window` for two-stage adaptive velocity-window refinement.

## Per-particle electric and guiding-center work

Build the electric-field interpolators once, then reuse them for saved in-memory
trajectories (Cartesian positions in m, velocities in m/s, times in s):

```julia
itp = build_field_work_interpolators()
summary = field_work(sol, itp; species = "O2+")
profile = Electric_field_work_profile(sol, itp; species = "O2+")
# profile.t, profile.conv_eV, profile.hall_eV, profile.gradpe_eV, profile.res_eV, profile.total_eV
# profile.delta_kinetic_eV, profile.energy_residual_eV
# profile.power_conv_eV_s, profile.power_hall_eV_s, profile.power_gradpe_eV_s, profile.power_res_eV_s

# Guiding-center / adiabatic work channels and adiabaticity (TestParticle v0.26)
ad = adiabatic_work_profile(sol, fields; species = "O2+")
# ad.parallel_eV, ad.fermi_eV, ad.grad_eV, ad.betatron_local_eV, ad.betatron_eV, ad.kappa

# solutions: TestParticle ensemble or vector returned by trace_forward
per_particle = particle_field_work(solutions, itp; threaded = true)
histories = particle_field_work(solutions, itp; profiles = true, threaded = true)
```

Each component is signed work `q ∫ E_component ⋅ v dt` evaluated on the **same
total-field trajectory** (including convective $-\mathbf{U}\times\mathbf{B}$, Hall $(\mathbf{J}\times\mathbf{B})/(n_e e)$, electron pressure gradient $-\nabla p_e/(n_e e)$, and resistive $\eta\mu_0\mathbf{J}$ channels when present in the `.vts` file). Positive work adds energy; negative work removes it.
Power is in eV/s, cumulative profiles in eV; `field_work` also returns J.
The work functions do not run trajectories or save files. Single-particle calls
reject ensembles with multiple members to avoid silently using only the first.

The residual `ΔK - W_total` diagnoses numerical energy closure for nonrelativistic
trajectories subject only to electromagnetic forces. Check timestep and saved
sample cadence: quadrature cannot recover missing trajectory structure. Reverse
time integrations preserve signed time intervals. Work fractions do not predict
the result of disabling a field component, because doing so changes the trajectory.
Avoid evaluating interpolators beyond their field domain.

Run the unit tests:

```sh
julia --threads=2 --project=. test/runtests.jl
```

The former name `field_work_profile` remains available as a compatibility alias for
`Electric_field_work_profile`, with identical arguments and return values.

`examples/forward_tracing/hemisphere_work.jl` demonstrates `Electric_field_work_profile` on a
reproducible 1000-particle, 800 km O2+ shell release. It uses four
threads when launched by `examples/forward_tracing/plot_hemisphere_work.jl`, starts at rest, and stops
at the 200 km or 4 Mars-radius boundary (20,000 s guard). Work ends at the last
valid in-domain state. Energy closure above 0.1% triggers timestep refinement;
this is a diagnostic, not a guarantee of full trajectory convergence.

Set `HEMISPHERE_WORK_PREVIEW` to customize the output PNG path. `examples/forward_tracing/plot_hemisphere_work.jl` plots six
XZ panels with dayside and nightside particles in separate rows and convection,
Hall and total work in three columns using `PyPlot`. Each row uses its own symmetric-log color
scale. See [the example documentation](examples/README.md). Only the preview image is
saved; trajectory data pass through memory. Threaded output is not ordered, so
each record carries its original particle ID.

## Scripts and Examples Overview

- **Data Conversion & Diagnostics (`scripts/`)**:
  - `scripts/convert_batsrus_mhd.jl`: Convert BATSRUS `.dat` files into `.vts` and `.mat` source-rate files.
  - `scripts/single_particle_work.jl` & `scripts/plot_single_particle_work.jl`: Single-particle forward tracing, step-convergence check, and cumulative work/power plotting.
  - `scripts/dayside_work.jl` & `scripts/plot_dayside_work.jl`: Dayside 800 km shell work integration and plotting.
  - `scripts/sphere_trajectories.jl` & `scripts/plot_dayside_yz.jl`: Spherical shell trajectory tracing and dayside YZ projections.
  - `scripts/random_probe_backtrace.jl`, `scripts/check_backtrace_probe.jl`, `scripts/smoke_ionosphere.jl`: Backtrace probe checks and ionosphere smoke tests.
- **Background Models (`examples/background_models/`)**:
  - `plot_atmosphere.jl`: GITM neutral density/temperature and AMPS hot oxygen profiles and maps.
  - `plot_ionosphere_maps.jl`: MHD O2+ flux and temperature maps at 200 and 400 km.
  - `plot_outflow_500km.jl`, `plot_outflow_altitudes.jl`, `plot_speed_altitudes.jl`, `plot_speed_inward_altitudes.jl`, `plot_speed_outward_altitudes.jl`: Multi-altitude O2+ and O+ flux/speed maps.
- **Forward Tracing (`examples/forward_tracing/`)**:
  - `sphere_trajectories.jl`, `plot_sphere_trajectories.jl`, `plot_trajectories_xz_yz.jl`: 800 km spherical shell release trajectories in XZ/YZ planes.
  - `hemisphere_work.jl`, `plot_hemisphere_work.jl`: Dayside and nightside convective, Hall, and total work maps.
  - `monte_carlo_shell.jl`, `analyze_monte_carlo.jl`, `maxwellian_source.jl`: Weighted Maxwellian shell Monte Carlo tracing and analysis.
  - `monte_carlo_forward_tracing/`: Full shell Monte Carlo suite (`monte_carlo_shell.jl`, `run_incremental_chunks.jl`, `analyze_paper_results.jl`, `analyze_saved_probes.jl`, `plot_monte_carlo_sampling.jl`, `plot_bulk_speed_flux_maps.jl`, `plot_trajectories.jl`, `plot_library_psd.jl`, `satellite_orbit_distributions.jl`).
- **Backward Tracing (`examples/backward_tracing/`)**:
  - `random_probe_backtrace.jl`, `plot_probe_backtrace.jl`: Random-velocity backtrace trajectories from `(0, 0, 2 Rm)`.
  - `probe_vdf_integrated.jl`, `plot_probe_vdf.jl`, `check_probe_vdf_grid.jl`: Velocity-integrated backtraced VDFs.
  - `probe_energy_gain.jl`, `plot_energy_gain.jl`, `refine_energy_gain.jl`: PSD-weighted forward-time energy gain maps.
  - `probe_path_power.jl`, `plot_path_power.jl`: Local power and cumulative work along backtraced paths.

## Perlmutter

Use `scripts/perlmutter/run_backtrace_vdf.slurm` as the starting point. Copy or
generate the MHD `.vts` file into `data/` on Perlmutter before submitting.

## MHD ionosphere thin-sheet source in backtracing

The source is a penetrable shell, **400 km by default**. The absorbing inner
boundary is always **200 km**, independent of source altitude (allowed range:
200 to 800 km). Backtracing continues below the shell and accumulates volume
production until it reaches 200 km, the outer boundary, or the time limit.

```julia
cfg = BacktraceConfig(ionosphere_altitude_km=400.0, dt=-0.05)
result = run_backtrace_vdf(cfg)
```

MHD O2+ density, ion temperature and Cartesian bulk velocity define
`F = n*norm(Ui)` in m^-2 s^-1 and a normalized drifting Maxwellian `g` in
s^3 m^-3. Each transverse crossing adds `F*g/abs(dot(v,er))` in s^3 m^-6,
where `v` is the traced particle velocity. All resolved crossings count, in
either direction. The source shell does not terminate or scatter particles.
This replaces the previous density-normalized boundary-VDF prescription.

Volume production still uses the MAT volume rate and a zero-drift
neutral-temperature Maxwellian. The saved `f2d_volume` and `f2d_ionosphere`
are integrated over Vy in s^2 m^-5; `f2d_xz` is their sum.
`ionosphere_crossings` counts sheet encounters per Vx/Vz cell across sampled Vy.
`model="thin_shell_source_v1"` distinguishes these files from older results.

`load_ionosphere_source`, `ionosphere_properties` and `ionosphere_distribution`
expose the MHD shell moments. The last helper returns both normalized `g` and
`f=n*g`; the transport source uses `flux*g`, not `flux*f`.
`include_ionosphere=false` disables only the sheet source, retaining 200 km
termination. Detectors below the source sheet are supported. Detectors exactly
on the ideal sheet and unresolved grazing crossings raise errors; no arbitrary
radial-speed floor is applied. A finite-thickness model would be required to
regularize these cases.

See the [full derivation and unit audit](examples/backward_tracing/backtracing_derivation.md)
for the transport equation, delta-function change of variables, volume
quadrature, crossing treatment, unit table, background assumptions and limits.
Run `julia --project=. test/runtests.jl` for analytic tests and
`julia --project=. scripts/smoke_ionosphere.jl` for the local MHD smoke test.

## Trajectory visualization

[src/visualization](src/visualization/README.md) provides Julia (`PyPlot`) plots of Mars, BS/MPB and two- or three-dimensional trajectories (`plot_mars`, `bs_mpb`, `plot_mars_context`, `load_trajectories`, `plot_trajectory`), with species and particle-ID selection from forward or backward trajectory files.


## Save forward electric work and compact trajectory archiving

Pass the same-grid field interpolators when writing each trajectory batch:

```julia
itp = build_field_work_interpolators(; include_extended = true) # includes gradpe & resistive
write_trajectory_batch("outputs/new_run/trajectories_00001.jld2", solutions;
    particle_ids=collect(1:length(solutions)), rate_weights_s=rate_weights_s,
    work_itp=itp, work_mode=:steps, save_power=false,
    packed=true, state_eltype=Float32)
foreach_saved_trajectory("outputs/new_run/trajectories_00001.jld2") do r
    dw = r.work.steps["hall"]       # eV, N-1 intervals ("total", "conv", "hall", "gradpe")
    cumulative = vcat(0., cumsum(dw))
    mean_power = dw ./ diff(r.trajectory.t) # eV/s, actual interval duration
    summary = r.work.summary
end
```

Use `work_mode=:summary` for signed totals, positive/negative interval sums,
initial/final kinetic energy, final and maximum absolute energy-closure residuals,
and final field-sum residual, without allocating interval-work histories.
`save_power=true` additionally saves instantaneous endpoint power (N samples).
Default `:none` preserves existing writer behavior. `foreach_saved_trajectory` and
`load_trajectories` transparently read format 3 (`packed=true`), format 2, format 1,
and legacy files (`r.work === nothing`).

### Compact Storage Options (`ShellMonteCarlo.Config` & `write_trajectory_batch`)

In `examples/forward_tracing/monte_carlo_forward_tracing/monte_carlo_shell.jl`, `ShellMonteCarlo.Config` supports the following storage and physics options for large-scale local and cluster runs:

- **`packed_trajectories::Bool = true` (`packed=true` in `write_trajectory_batch`, format v3)**:
  Packs all trajectories in each batch into a single contiguous `6 × TotalPoints` matrix (`states`: `x, y, z, vx, vy, vz`) and `n_comp × TotalIntervals` work matrix (`work/dW_eV`) indexed by `point_offsets`, eliminating per-particle HDF5/JLD2 group metadata overhead.
- **`trajectory_float32::Bool = true` (`state_eltype=Float32` in `write_trajectory_batch`)**:
  Stores saved trajectory positions, velocities, and step work increments in `Float32` while keeping full `Float64` precision during integration and summary accumulation.
- **Dropped Redundant `t` Array**:
  Format v3 omits the per-step `t` vector for uniform-step trajectories and instead stores only three scalars per particle (`t_start_s`, `dt_step_s`, and exact boundary-clipped `t_end_s`), reconstructing exact `Float64` timestamps on read (`2.35x` smaller batch files than per-particle `Float64` groups).
- **`skip_inner_and_trapped::Bool = false`**:
  When `true`, skips writing `inner` (absorbed at 200 km) and low-altitude trapped (`time_limit`) particles to `particles.csv` and `trajectories_*.jld2` (while still recording exact global termination counts in `completion.toml`). Since `time_limit` and `inner` particles account for ~85.7% of all integration steps without reaching magnetospheric detectors, skipping them reduces storage by ~7x.
- **`save_trajectories::Symbol = :all`**:
  Controls which trajectories are written to `trajectories_*.jld2`:
  - `:all` — save all trajectories (or all `outer`/probe trajectories when `skip_inner_and_trapped=true`).
  - `:outer_only` — save all escaping (`outer`) trajectories plus any probe hits (`~6.8 GB` per $10^6$ launched particles at `save_stride=5`).
  - `:roi_and_sample` — save only trajectories intersecting the paper's virtual spacecraft detector tubes (circular $2\,R_M$ orbit and cross-tail current sheet transect) plus an unbiased random sample of escaping trajectories (`trajectory_sample_size = 20_000`, recorded in `plot_sample_ids.toml`) (`~1.6 GB` per $10^6$ launched particles at `save_stride=5`).
  - `:probes_only` — save only trajectories crossing the primary detector cube.
  - `:none` — save `particles.csv` and probe files only.
- **`save_stride::Int = 1`**:
  Subsamples saved trajectory points by `save_stride` (always retaining the exact initial and terminal boundary states) after full-cadence tracing.
- **`fields_path::String = ""` & `include_gradpe_work::Bool = false`**:
  Selects a custom `.vts` field file (e.g., `data/mars_fields_with_crustal.vts` or `data/mars_fields_without_crustal.vts`) and enables 4-component Ohm's-law work tracking (`total`, `conv`, `hall`, `gradpe`).
- **`chunk_id::Int = 1` (Incremental Multi-Chunk Accumulation)**:
  Assigns a strictly disjoint per-cell RNG stream (`seed + (chunk_id - 1) * seed_stride + cellid`) and non-overlapping global particle IDs (`(chunk_id - 1) * N_chunk + 1 .. chunk_id * N_chunk`) to each chunk. Passing a vector of chunk directories `["outputs/mc_chunk_01", "outputs/mc_chunk_02", ...]` to `foreach_saved_trajectory`, `forward_psd_saved`, `load_trajectories`, or `run_paper_analysis` validates that all `(seed, chunk_id)` pairs are distinct and automatically scales single-chunk rate weights by `1 / K`.

To run incremental chunks locally and monitor chunk-to-chunk statistical convergence (`*_convergence.toml` and `*_convergence.png`):

```sh
julia --threads=8 --project=. examples/forward_tracing/monte_carlo_forward_tracing/run_incremental_chunks.jl outputs/mc_400km_crustal 3 100
```

To reproduce the paper's virtual spacecraft and cumulative work figures (`Figures 3–7`) from one or more saved chunk directories:

```sh
julia --threads=8 --project=. examples/forward_tracing/monte_carlo_forward_tracing/analyze_paper_results.jl outputs/mc_400km_crustal_chunk_01 outputs/mc_400km_crustal_chunk_02 outputs/mc_400km_crustal_cum_02_paper
```
