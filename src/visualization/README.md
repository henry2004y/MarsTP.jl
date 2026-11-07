# Mars boundaries and trajectory visualization

Julia visualization using `PyPlot` (Matplotlib), `JLD2`, `JSON`, and `TOML`. No `py_space_zc` installation is required. Run commands from the repository root with `julia --project=.`.

## Trajectory plots

`plot_trajectory` reads saved trajectories and uses `mars.jl` to draw Mars, the bow shock (BS), and the magnetic pileup boundary (MPB). Output is PNG.

### Julia interface

```julia
using MarsTP
using PyPlot

fig, axes, records = plot_trajectory(
    "outputs/my_run";
    species = "O2+",
    count = 100,
    planes = ("XZ", "3D"),
    output = "outputs/trajectory.png",
)
```

### Command-line usage

```powershell
# Forward JLD2 batches: select O2+ and draw three projections.
julia --project=. -e "using MarsTP, PyPlot; fig, _, recs = plot_trajectory(\`"outputs/my_run\`"; species=\`"O2+\`", count=5000, output=\`"outputs/forward.png\`"); PyPlot.plt.close(fig)"

# Backward JSONL: positions in Rm and times in s.
julia --project=. -e "using MarsTP, PyPlot; fig, _, recs = plot_trajectory(\`"outputs/my_backtrace/trajectories.jsonl\`"; position_unit=\`"Rm\`", species=\`"O2+\`", planes=(\`"XZ\`", \`"XY\`", \`"YZ\`", \`"3D\`"), output=\`"outputs/backward.png\`"); PyPlot.plt.close(fig)"
```

`species` selects the ion species; `particle_ids` selects saved particle numbers. Species labels are read from per-particle fields, file fields, or adjacent `metadata.toml`/`metadata.json`. Mixed-species data can label individual records. For old data without labels, `assume_species = "O2+"` explicitly declares the species. It does not override existing labels or transform particles into another species.

`count` limits displayed trajectories (default 5000), using uniform reservoir sampling with a fixed seed. `max_points` limits displayed points per trajectory (default 300), retaining both endpoints. A file and particle ID together identify a path; pass a single file to narrow the selection.

### File formats

- MarsTP JLD2 batches: `p<ID>/state` is `7×N` (or `N×7`), with `t, x, y, z, vx, vy, vz` in s, m, and m/s. Increasing and decreasing time sequences are retained for forward and backward paths.
- JSONL: each object requires `id`, `N×3` `points`, and `N` `times_s` values. Optional fields include `species`, `coordinate_system`, and `position_unit`. Position units are `"m"`, `"km"`, or `"Rm"`, optionally specified with `position_unit`.
- PSD-only or endpoint-only files cannot reconstruct full paths; save trajectories during integration.

The default Mars radius is 3,390,000 m, configurable with `rm_m`. Plot coordinates use `Rm`. For JLD2, only selected trajectories and display points are read into memory.

## Mars, BS and MPB

In two dimensions, `plot_mars` defaults to the bundled [Mars image](mars_globe_true_color.png). No image path is required. Use `texture = false` for a solid disk or `texture_path = ...` for a custom image.

Three-dimensional axes display a sphere. The bundled disk photograph is not a global longitude/latitude texture and is not wrapped onto the sphere.

`bs_mpb` uses empirical conic parameters. Let $x, \rho, r, x_0, L$ be in $R_M$, $\theta$ in rad, and eccentricity $\epsilon$ dimensionless:

$$
r=\frac{L}{1+\epsilon\cos\theta},\qquad
x=x_0+r\cos\theta,\qquad \rho=r\sin\theta.
$$

| Boundary | $x_0$ ($R_M$) | $L$ ($R_M$) | $\epsilon$ | Region |
| --- | ---: | ---: | ---: | --- |
| BS | 0.600 | 2.081 | 1.026 | All |
| Dayside MPB | 0.640 | 1.080 | 0.770 | $x \ge 0$ |
| Nightside MPB | 1.600 | 0.528 | 1.009 | $x < 0$ |

Only positive-radius branches are used. Rotation about $X$ generates the three-dimensional surfaces; XY and XZ show symmetric sections. For YZ, `x_slice = 0` displays the boundary section at $X = 0\,R_M$. No YZ boundary is drawn by default.

The model assumes $+X$ sunward. For other coordinate conventions, transform positions first or use `boundaries = false`.

```julia
using MarsTP
using PyPlot

fig, ax = PyPlot.plt.subplots()
plot_mars_context(ax, "XZ"; xmin = -4.2)
ax.set(xlim = (-4.2, 4.2), ylim = (-4.2, 4.2), aspect = "equal")

fig = PyPlot.plt.figure()
ax = fig.add_subplot(projection = "3d")
bs_mpb(ax, "3D")
plot_mars(ax)
ax.set_box_aspect((1, 1, 1))
```
