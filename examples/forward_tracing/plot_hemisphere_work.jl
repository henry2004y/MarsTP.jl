# Day/night work maps: shared scales within rows, independent scales between rows using MarsTP and PyPlot.
using MarsTP
using JSON
using PyPlot

const ROOT = project_path()
const PREVIEW = get(ENV, "HEMISPHERE_WORK_PREVIEW", joinpath(@__DIR__, "images", "electric_work_800km.png"))
mkpath(dirname(abspath(PREVIEW)))

cmd = setenv(
    `$(Base.julia_cmd()) --startup-file=no --threads=auto --project=$ROOT $(joinpath(@__DIR__, "hemisphere_work.jl"))`,
    "WORK_HEMISPHERE" => "all"
)
records = [JSON.parse(line) for line in eachline(cmd) if startswith(strip(line), "{")]

dayside = [r for r in records if r["points"][1][1] > 0.0]
nightside = [r for r in records if r["points"][1][1] <= 0.0]
for (group, name) in ((dayside, "Dayside"), (nightside, "Nightside"))
    max_rel_res = maximum(abs(Float64(r["summary"][4])) / max(abs(Float64(r["summary"][3])), 1.0) for r in group) * 100.0
    max_work_kev = maximum(maximum(abs, Float64.(pt[3:5])) for r in group for pt in r["points"]) / 1000.0
    println("$name work ($(length(group))): max |W| = $(round(max_work_kev, digits = 4)) keV, max rel residual = $(round(max_rel_res, digits = 4)) %")
end

fig, axes = PyPlot.plt.subplots(2, 3; figsize = (16, 10), layout = "constrained")
for (row, (group, name)) in enumerate(zip((dayside, nightside), ("Dayside", "Nightside")))
    arrays = [reduce(vcat, [Float64.(pt)' for pt in r["points"]]) for r in group]
    limit = maximum(maximum(abs, @view a[:, 3:5]) for a in arrays) / 1000.0
    norm = PyPlot.matplotlib.colors.SymLogNorm(linthresh = 0.1, vmin = -limit, vmax = limit, base = 10)
    segs = Matrix{Float64}[]
    for a in arrays, i in 1:(size(a, 1) - 1)
        push!(segs, [a[i, 1] a[i, 2]; a[i + 1, 1] a[i + 1, 2]])
    end
    lines_lc = nothing
    for (col, title) in enumerate(("Convection work", "Hall work", "Total work"))
        ax = axes[row, col]
        bs_mpb(ax, "XZ"; sphere = false, boundary_color = "0.4")
        plot_mars(ax; alpha = 0.5, zorder = 1)
        vals = Float64[]
        for a in arrays, i in 1:(size(a, 1) - 1)
            push!(vals, 0.5 * (a[i, col + 2] + a[i + 1, col + 2]) / 1000.0)
        end
        lines_lc = PyPlot.matplotlib.collections.LineCollection(segs; array = vals, cmap = "RdBu_r", norm = norm, linewidths = 0.6, alpha = 0.85, zorder = 3)
        ax.add_collection(lines_lc)
        ax.set(
            xlim = (-4.15, 4.15), ylim = (-4.15, 4.15), aspect = "equal",
            xlabel = "X / R_M", ylabel = (col == 1 ? "$name\nZ / R_M" : "Z / R_M"),
            title = row == 1 ? title : ""
        )
        ax.grid(alpha = 0.12)
    end
    fig.colorbar(lines_lc; ax = collect(axes[row, :]), location = "right", fraction = 0.025, pad = 0.02, label = "Cumulative work (keV)")
end
fig.savefig(PREVIEW; dpi = 170)
PyPlot.plt.close(fig)
println("Preview: ", PREVIEW)
