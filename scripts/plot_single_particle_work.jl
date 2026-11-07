# Display single-particle trajectory and signed field-work diagnostics using MarsTP and PyPlot.
using MarsTP
using DelimitedFiles
using PyPlot

const ROOT = project_path()
const PREVIEW = get(ENV, "WORK_PREVIEW", project_path("outputs", "single_particle_work.png"))
mkpath(dirname(abspath(PREVIEW)))

out_str = read(`$(Base.julia_cmd()) --startup-file=no --project=$ROOT $(joinpath(@__DIR__, "single_particle_work.jl"))`, String)
lines = split(strip(out_str), '\n')
header = split(strip(lines[1]), ",")
mat = readdlm(IOBuffer(join(lines[2:end], "\n")), ',', Float64)
d = Dict(String(header[i]) => mat[:, i] for i in eachindex(header))

fig, axes = PyPlot.plt.subplots(1, 3; figsize = (16, 5), layout = "constrained")
ax = axes[1]
bs_mpb(ax, "XZ"; sphere = false, boundary_color = "0.4")
plot_mars(ax; alpha = 0.55, zorder = 1)
ax.plot(d["x"], d["z"]; color = "#16749a", lw = 1.7, zorder = 3)
ax.scatter([d["x"][1]], [d["z"][1]]; s = 40, c = "#339354", label = "Start", zorder = 4)
ax.scatter([d["x"][end]], [d["z"][end]]; s = 40, c = "#bd432c", label = "End", zorder = 4)
ax.set(xlim = (-2.0, 2.0), ylim = (-0.5, 4.2), aspect = "equal", xlabel = "X / R_M", ylabel = "Z / R_M", title = "Dayside O2+")
ax.legend(frameon = false, fontsize = 9)

ax = axes[2]
ax.plot(d["t"], d["Wconv"]; label = "Convection work", color = "#16749a")
ax.plot(d["t"], d["Whall"]; label = "Hall work", color = "#b46924")
ax.plot(d["t"], d["Wtotal"]; label = "Total work", color = "#9560a8", lw = 2.5, alpha = 0.6)
ax.plot(d["t"], d["K"] .- d["K"][1]; label = "Kinetic energy change", color = "black", ls = "--", lw = 1.0)
ax.axhline(0.0; color = "0.6", lw = 0.6)
ax.set(xlabel = "Time (s)", ylabel = "Energy (eV)", title = "Cumulative work")
ax.legend(frameon = false, fontsize = 9)

ax = axes[3]
ax.plot(d["t"], d["Pconv"]; label = "Convection", color = "#16749a")
ax.plot(d["t"], d["Phall"]; label = "Hall", color = "#b46924")
ax.axhline(0.0; color = "0.6", lw = 0.6)
ax.set(xlabel = "Time (s)", ylabel = "Power (eV/s)", title = "Instantaneous power")
ax.legend(frameon = false, fontsize = 9)

for a in axes
    a.grid(alpha = 0.15)
end
fig.savefig(PREVIEW; dpi = 170)
PyPlot.plt.close(fig)
println("Preview: ", PREVIEW)

