# XZ trajectory segments colored by physical forward-time electric power using MarsTP and PyPlot.
using MarsTP
using DelimitedFiles
using JSON
using PyPlot

function read_named_csv(path)
    lines = readlines(path)
    header = split(strip(lines[1]), ",")
    data = readdlm(IOBuffer(join(lines[2:end], "\n")), ',', Float64)
    return Dict(String(header[i]) => data[:, i] for i in eachindex(header))
end

out = abspath(ARGS[1])
case_path = joinpath(out, "case.json")
case = isfile(case_path) ? JSON.parsefile(case_path) : Dict("mode" => "original", "detector_Rm" => [0.0, 0.0, 2.0], "dt_s" => -0.05)
probe = Float64.(case["detector_Rm"])
tail = String(case["mode"]) != "original"
image_dir = joinpath(@__DIR__, "images")
mkpath(image_dir)

a = read_named_csv(joinpath(out, "segments.csv"))
b = read_named_csv(joinpath(out, "particles.csv"))
nseg = length(a["id"])
ids = round.(Int, a["id"])

segs = [ [a["x0_Rm"][i] a["z0_Rm"][i]; a["x1_Rm"][i] a["z1_Rm"][i]] for i in 1:nseg ]
values = [a["conv_eV_s"], a["hall_eV_s"], a["total_eV_s"]]
cumulative = Vector{Float64}[]
for (comp, power) in zip(("conv", "hall", "total"), values)
    after = a["lookback_$(comp)_eV"]
    dw = power .* a["elapsed_s"]
    push!(cumulative, b["$(comp)_eV"][ids] .- after .+ 0.5 .* dw)
end

limits = [10.0^ceil(log10(maximum(maximum(abs, v) for v in row))) for row in (values, cumulative)]
cmap = PyPlot.plt.get_cmap("coolwarm")
fig, axs = PyPlot.plt.subplots(2, 3; figsize = (183 / 25.4, 177 / 25.4), layout = "constrained", sharey = true)

for (row, rowvalues) in enumerate((values, cumulative))
    norm = PyPlot.matplotlib.colors.SymLogNorm(linthresh = 1.0, vmin = -limits[row], vmax = limits[row], base = 10)
    for (col, (v, title)) in enumerate(zip(rowvalues, ("Convection", "Hall", "Total")))
        ax = axs[row, col]
        bs_mpb(ax, "XZ"; draw_mpb = false, sphere = false, boundary_color = "#444444", boundary_ls = "--", boundary_lw = 0.6)
        bs_mpb(ax, "XZ"; draw_bs = false, sphere = false, boundary_color = "#444444", boundary_ls = ":", boundary_lw = 0.6)
        lc = PyPlot.matplotlib.collections.LineCollection(segs; array = v, cmap = cmap, norm = norm, linewidths = 0.20, alpha = 0.7, rasterized = true)
        ax.add_collection(lc)
        plot_mars(ax; radius = 1.0, alpha = 0.95, zorder = 3)
        for radius in (Rinner / Rm, 4.0)
            ax.add_patch(PyPlot.matplotlib.patches.Circle((0.0, 0.0), radius; fill = false, ls = "--", lw = 0.5, color = "#888888"))
        end
        ax.scatter([probe[1]], [probe[3]]; marker = "*", s = 45, c = "black", edgecolors = "white", linewidths = 0.7, zorder = 5)
        ax.set(xlim = (-4.2, 4.2), ylim = (-4.2, 4.2), aspect = "equal", xlabel = "X / R_M", title = title)
    end
    axs[row, 1].set_ylabel("Z / R_M")
    label = row == 1 ? "Local power (eV/s)" : "Cumulative work from past endpoint (eV)"
    sm = PyPlot.matplotlib.cm.ScalarMappable(norm = norm, cmap = cmap)
    fig.colorbar(sm; ax = collect(axs[row, :]), location = "bottom", shrink = 0.85, pad = 0.06, label = label)
end

fig.suptitle("$(length(b["id"])) O2+ backtraces at ($(probe[1]), $(probe[2]), $(probe[3])) R_M\nLocal power and cumulative work"; fontsize = 10)
image = joinpath(image_dir, tail ? "tail_probe_path_power_xz_5000.png" : "probe_path_power_xz_5000.png")
fig.savefig(image; dpi = 400)
PyPlot.plt.close(fig)
println("Saved ", image)

