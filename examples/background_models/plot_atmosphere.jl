# Plot prepared GITM/AMPS inputs and the MarsTP neutral atmosphere extension using PyPlot.
using MarsTP
using MAT
using JSON
using Printf
using PyPlot

const ROOT = project_path()
const OUT = @__DIR__
mkpath(joinpath(OUT, "images"))

gitm = load_gitm()
amps = load_amps()
gitm_raw = matread(data_path("gitm_sph.mat"))
amps_raw = matread(data_path("amps_sph.mat"))

heights_km = vcat(collect(100.0:5.0:220.0), collect(225.0:5.0:600.0))
nCO2_prof = Float64[]
nO_prof = Float64[]
Tn_prof = Float64[]
nO_hot_prof = Float64[]

for h in heights_km
    r = Rm + h * 1.0e3
    np_val = neutral_properties(gitm, r, 0.5 * pi, 0.0)
    push!(nCO2_prof, np_val.nCO2 / 1.0e6)
    push!(nO_prof, np_val.nO / 1.0e6)
    push!(Tn_prof, np_val.Tn)
    push!(nO_hot_prof, hot_oxygen_density(amps, r, 0.5 * pi, 0.0) / 1.0e6)
end

fig, axes = PyPlot.plt.subplots(3, 2; figsize = (9.2, 10.5), layout = "constrained")
axn, axt = axes[1, 1], axes[1, 2]

mask_lo = heights_km .<= 220.0
mask_hi = heights_km .>= 220.0

axn.plot(nCO2_prof[mask_lo], heights_km[mask_lo], "-"; color = "#247b98", lw = 1.6, label = "GITM CO2")
axn.plot(nCO2_prof[mask_hi], heights_km[mask_hi], "--"; color = "#247b98", lw = 1.6)
axn.plot(nO_prof[mask_lo], heights_km[mask_lo], "-"; color = "#bb6b25", lw = 1.6, label = "GITM O")
axn.plot(nO_prof[mask_hi], heights_km[mask_hi], "--"; color = "#bb6b25", lw = 1.6)
axn.plot(nO_hot_prof, heights_km, "-"; color = "#79549e", lw = 1.6, label = "AMPS hot O")

axt.plot(Tn_prof[mask_lo], heights_km[mask_lo], "-"; color = "#247b98", lw = 1.6)
axt.plot(Tn_prof[mask_hi], heights_km[mask_hi], "--"; color = "#247b98", lw = 1.6)

axn.set_xscale("log")
axn.set_xlabel("Number density (cm^-3)")
axt.set_xlabel("Temperature (K)")
axn.legend(loc = "lower left", fontsize = 8)

for (ax, title) in ((axn, "a  Density"), (axt, "b  Neutral temperature"))
    ax.set_title(title; loc = "left", fontweight = "bold")
    ax.set_ylabel("Altitude (km)")
    ax.set_ylim(100, 600)
    ax.axhline(220; color = "0.65", ls = ":", lw = 0.8)
    ax.grid(alpha = 0.15)
end

panels = (
    (axes[2, 1], "c  GITM CO2 (200 km)", gitm_raw, "nCO2", 200.0e3, 1.0e6, true),
    (axes[2, 2], "d  GITM O (200 km)", gitm_raw, "nO", 200.0e3, 1.0e6, true),
    (axes[3, 1], "e  GITM neutral temperature (200 km)", gitm_raw, "Tn", 200.0e3, 1.0, false),
    (axes[3, 2], "f  AMPS hot O (500 km)", amps_raw, "nO_hot", 500.0e3, 1.0e6, true),
)

for (ax, title, raw, key, alt_m, scale, is_log) in panels
    alt_vec = vec(Float64.(raw["altitude"]))
    lat_vec = vec(Float64.(raw["latitude"]))
    lon_vec = vec(Float64.(raw["longitude"]))
    ih = argmin(abs.(alt_vec .- alt_m))
    slice_2d = Float64.(raw[key][ih, :, :]) ./ scale
    norm = is_log ?
        PyPlot.matplotlib.colors.LogNorm(minimum(slice_2d), maximum(slice_2d)) :
        PyPlot.matplotlib.colors.Normalize(minimum(slice_2d), maximum(slice_2d))
    mesh = ax.pcolormesh(lon_vec, lat_vec, slice_2d; shading = "nearest", cmap = "turbo", norm = norm, rasterized = true)
    ax.set_title(title; loc = "left", fontweight = "bold")
    ax.set(xlim = (0, 360), ylim = (-90, 90), xlabel = "Longitude (deg)", ylabel = "Latitude (deg)")
    cb = fig.colorbar(mesh; ax = ax, pad = 0.025, fraction = 0.045, aspect = 24)
    cb.set_label(is_log ? "Density (cm^-3)" : "Temperature (K)"; fontsize = 9)
end

fig.suptitle("GITM and AMPS atmosphere"; fontsize = 15, fontweight = "bold")
out_png = joinpath(OUT, "images", "atmosphere_gitm200km_amps500km.png")
fig.savefig(out_png; dpi = 300)
PyPlot.plt.close(fig)
println("Saved ", out_png)

