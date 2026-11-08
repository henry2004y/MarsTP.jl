# Publication-quality script for Figure 3 of the Mars O2+ test-particle manuscript:
#   Left column (O2+ trajectories, 20,000 paths in blue-ish colormap):
#     (a) XZ projection with BS (dashed), MPB (solid), Mars disk, Detector star at (-2, 0, 0) Rm,
#         "Plume" label, and E_sw upward arrow
#     (b) XY projection with BS, MPB, Mars disk, and Detector star
#     (c) YZ projection with circular BS & MPB cross-sections at X = 0, Mars disk, and Detector star
#   Right column (Local distributions at (-2, 0, 0) Rm):
#     (d) Reduced VDF ∫f dv_y [cm^-3 (km s^-1)^-2] in (v_x, v_z) on [-50, 50] km/s, black background,
#         white dashed crosshairs, and U_bulk, T, B annotations
#     (e) 1D omnidirectional energy spectrum PSD (s^3 m^-6) vs Energy (eV) step plot
#     (f) 2D Energy (eV) vs Pitch angle (deg) PSD (s^3 m^-6) on black background
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig3.jl [PAPER_DIR] [OUTPUT_PNG] [RAW_RUN_DIR]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

function _load_sampled_trajectories(paper_dir::AbstractString, raw_run_dir = nothing; count::Int = 20_000)
    sample_jld = joinpath(paper_dir, "trajectories_20000_sample.jld2")
    if isfile(sample_jld)
        return jldopen(sample_jld, "r") do f
            pts = f["points_Rm"]
            [Matrix{Float64}(p) for p in pts]
        end
    end
    if raw_run_dir !== nothing && (isdir(raw_run_dir) || (raw_run_dir isa AbstractVector && !isempty(raw_run_dir)))
        recs = load_trajectories(raw_run_dir; species = "O2+", count = count, max_points = 200, seed = 20260906)
        return [Matrix{Float64}(r["points"]) for r in recs]
    end
    return Matrix{Float64}[]
end

function _draw_blue_trajectories!(ax, paths::Vector{Matrix{Float64}}, plane::String)
    isempty(paths) && return nothing
    i, j = plane == "XZ" ? (1, 3) : plane == "XY" ? (1, 2) : (2, 3)
    nrec = length(paths)
    cmap_blues = plt.get_cmap("Blues")
    colors = [cmap_blues(0.48 + 0.45 * (k - 1) / max(nrec - 1, 1)) for k in 1:nrec]
    segments = Vector{Matrix{Float64}}(undef, nrec)
    @inbounds for k in 1:nrec
        p = paths[k]
        m = size(p, 1)
        seg = Matrix{Float64}(undef, m, 2)
        for r in 1:m
            seg[r, 1] = p[r, i]
            seg[r, 2] = p[r, j]
        end
        segments[k] = seg
    end
    lc = matplotlib.collections.LineCollection(
        segments;
        colors = colors,
        linewidths = 0.22,
        alpha = 0.14,
        zorder = 2,
        rasterized = true,
    )
    ax.add_collection(lc)
    return lc
end

function _draw_mars_and_boundaries!(ax, plane::String; xmin::Real = -4.0)
    if plane in ("XZ", "XY")
        bs_mpb(ax, plane; draw_bs = true, draw_mpb = false, xmin = xmin, boundary_color = "black", boundary_ls = "--", boundary_lw = 1.1)
        bs_mpb(ax, plane; draw_bs = false, draw_mpb = true, xmin = xmin, boundary_color = "black", boundary_ls = "-", boundary_lw = 1.1)
    elseif plane == "YZ"
        bs_mpb(ax, "YZ"; draw_bs = true, draw_mpb = false, x_slice = 0.0, boundary_color = "black", boundary_ls = "--", boundary_lw = 1.1)
        bs_mpb(ax, "YZ"; draw_bs = false, draw_mpb = true, x_slice = 0.0, boundary_color = "black", boundary_ls = "-", boundary_lw = 1.1)
    end
    plot_mars(ax; radius = 1.0, zorder = 5)
    return ax
end

function plot_fig3(
        paper_dir::AbstractString = joinpath(pwd(), "outputs", "perlmutter_runs", "mc_400km_without_crustal_cum_005_paper"),
        out_png::AbstractString = joinpath(paper_dir, "fig3.png");
        raw_run_dir = nothing,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    d = jldopen(jld_path, "r") do f
        fxz = Matrix{Float64}(f["fig3_fxz_s2_m5"])
        vx_c = Vector{Float64}(f["fig3_vx_centers_ms"])
        vz_c = Vector{Float64}(f["fig3_vz_centers_ms"])
        vx_edges = haskey(f, "fig3_vx_edges_ms") ? Vector{Float64}(f["fig3_vx_edges_ms"]) : nothing
        vz_edges = haskey(f, "fig3_vz_edges_ms") ? Vector{Float64}(f["fig3_vz_edges_ms"]) : nothing
        ec = Vector{Float64}(f["energy_centers_eV"])
        edges = Vector{Float64}(f["energy_edges_eV"])
        def = Vector{Float64}(f["fig3_omni_def"])
        epa_wt = Matrix{Float64}(f["fig3_epa_wt"])
        pa_edges = Vector{Float64}(f["fig3_pitch_edges_deg"])

        mass = MarsTP.TP.SpeciesDict["O2+"].m
        qe = MarsTP.TP.qᵢ
        v_edges = sqrt.(2.0 .* edges .* qe ./ mass)
        d3v_shell = (4.0 * pi / 3.0) .* (v_edges[2:end] .^ 3 .- v_edges[1:(end - 1)] .^ 3)
        vol = (0.2 * Rm)^3

        psd_1d = if haskey(f, "fig3_psd_1d_s3_m6")
            Vector{Float64}(f["fig3_psd_1d_s3_m6"])
        else
            vec(sum(epa_wt; dims = 2)) ./ (vol .* d3v_shell)
        end

        epa_psd = if haskey(f, "fig3_epa_psd_s3_m6")
            Matrix{Float64}(f["fig3_epa_psd_s3_m6"])
        else
            mat = zeros(Float64, size(epa_wt))
            for ke in 1:size(epa_wt, 1), kp in 1:size(epa_wt, 2)
                dcos = cosd(pa_edges[kp]) - cosd(pa_edges[kp + 1])
                d3v_pa = (2.0 * pi / 3.0) * (v_edges[ke + 1]^3 - v_edges[ke]^3) * dcos
                mat[ke, kp] = epa_wt[ke, kp] / (vol * d3v_pa)
            end
            mat
        end

        u_bulk_kms = if haskey(f, "fig3_u_bulk_ms")
            Vector{Float64}(f["fig3_u_bulk_ms"]) ./ 1.0e3
        else
            dv = vx_c[2] - vx_c[1]
            n_tot = max(sum(fxz) * dv^2, 1.0e-30)
            ux = sum(fxz .* vx_c) * dv^2 / n_tot / 1.0e3
            uz = sum(fxz .* vz_c') * dv^2 / n_tot / 1.0e3
            [ux, 0.0, uz]
        end

        T_eV = if haskey(f, "fig3_T_eV")
            Float64(f["fig3_T_eV"])
        else
            dv = vx_c[2] - vx_c[1]
            n_tot = max(sum(fxz) * dv^2, 1.0e-30)
            vx2 = sum(fxz .* (vx_c .- u_bulk_kms[1] * 1.0e3) .^ 2) * dv^2 / n_tot
            vz2 = sum(fxz .* ((vz_c .- u_bulk_kms[3] * 1.0e3)') .^ 2) * dv^2 / n_tot
            mass * (vx2 + vz2) / (2.0 * qe)
        end

        b_nT = if haskey(f, "fig3_B_nT")
            Vector{Float64}(f["fig3_B_nT"])
        else
            [-0.12, 0.67, 0.00]
        end

        (; fxz, vx_c, vz_c, vx_edges, vz_edges, ec, edges, def, psd_1d, epa_psd, pa_edges, u_bulk_kms, T_eV, b_nT)
    end

    paths = _load_sampled_trajectories(paper_dir, raw_run_dir; count = 20_000)

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 14.5
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 14
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 14

    fig, axes = plt.subplots(3, 2; figsize = (9.9, 14.2), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.01, h_pad = 0.03, wspace = 0.01, hspace = 0.03)

    ax_a = axes[1, 1]
    ax_b = axes[2, 1]
    ax_c = axes[3, 1]
    ax_d = axes[1, 2]
    ax_e = axes[2, 2]
    ax_f = axes[3, 2]

    # Column headers
    ax_a.set_title("O\$_2^+\$ trajectories"; fontsize = 18.5, fontweight = "bold", pad = 16)
    ax_d.set_title("Local distributions\n(\$−2, 0, 0\$) \$R_M\$"; fontsize = 18.5, fontweight = "bold", pad = 12)

    # Left column: (a) XZ, (b) XY, (c) YZ
    for (ax, plane, tag, xlabel_str, ylabel_str, det_xy) in (
            (ax_a, "XZ", "(a)", "\$X_{MSE}\$ (\$R_M\$)", "\$Z_{MSE}\$ (\$R_M\$)", (-2.0, 0.0)),
            (ax_b, "XY", "(b)", "\$X_{MSE}\$ (\$R_M\$)", "\$Y_{MSE}\$ (\$R_M\$)", (-2.0, 0.0)),
            (ax_c, "YZ", "(c)", "\$Y_{MSE}\$ (\$R_M\$)", "\$Z_{MSE}\$ (\$R_M\$)", (0.0, 0.0)),
        )
        _draw_blue_trajectories!(ax, paths, plane)
        _draw_mars_and_boundaries!(ax, plane; xmin = -4.0)
        ax.scatter(
            [det_xy[1]], [det_xy[2]];
            marker = "*", s = 230, facecolor = "#ffd700", edgecolor = "black", lw = 1.0, zorder = 9,
        )
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal")
        ax.set_xticks(-4:2:4)
        ax.set_yticks(-4:2:4)
        ax.tick_params(labelsize = 14)
        ax.set_xlabel(xlabel_str; fontsize = 16)
        ax.set_ylabel(ylabel_str; fontsize = 16)
        ax.text(-0.16, 1.02, tag; transform = ax.transAxes, fontsize = 20, fontweight = "bold", va = "bottom", ha = "left")
    end

    # Annotations on panel a
    pe = PyPlot.PyCall.pyimport("matplotlib.patheffects")
    txt_det = ax_a.text(-2.0, 0.55, "Detector"; fontsize = 16, fontweight = "bold", ha = "center", va = "bottom", zorder = 10)
    txt_det.set_path_effects([pe.withStroke(linewidth = 2.8, foreground = "white")])
    ax_a.text(0.85, 2.55, "Plume"; fontsize = 15.5, ha = "center", va = "center", zorder = 10)
    ax_a.annotate(
        "", xy = (3.45, 2.6), xytext = (3.45, 0.8),
        arrowprops = Dict("arrowstyle" => "-|>", "color" => "black", "lw" => 2.0, "mutation_scale" => 16),
        zorder = 10,
    )
    ax_a.text(3.25, 1.7, "\$E_{sw}\$"; fontsize = 16.5, ha = "right", va = "center")

    # (d) Reduced VDF in Vx-Vz on [-50, 50] km/s
    ax_d.set_facecolor("black")
    fxz_plot = replace(permutedims(d.fxz), 0.0 => NaN)
    vx_km = d.vx_edges !== nothing ? d.vx_edges ./ 1.0e3 : d.vx_c ./ 1.0e3
    vz_km = d.vz_edges !== nothing ? d.vz_edges ./ 1.0e3 : d.vz_c ./ 1.0e3
    norm_d = matplotlib.colors.LogNorm(vmin = 1.0e-3, vmax = 1.0e0)
    im_d = ax_d.pcolormesh(vx_km, vz_km, fxz_plot; cmap = "turbo", norm = norm_d, shading = "auto", rasterized = true)
    ax_d.axhline(0.0; color = "white", ls = "--", lw = 1.0, alpha = 0.85)
    ax_d.axvline(0.0; color = "white", ls = "--", lw = 1.0, alpha = 0.85)
    ax_d.set_xlim(-50.0, 50.0)
    ax_d.set_ylim(-50.0, 50.0)
    ax_d.set_aspect("equal")
    ax_d.set_xticks(-50:25:50)
    ax_d.set_yticks(-50:25:50)
    ax_d.tick_params(labelsize = 14)
    ax_d.set_xlabel("\$v_x\$ (km s\$^{-1}\$)"; fontsize = 16)
    ax_d.set_ylabel("\$v_z\$ (km s\$^{-1}\$)"; fontsize = 16)
    ax_d.text(-0.18, 1.02, "(d)"; transform = ax_d.transAxes, fontsize = 20, fontweight = "bold", va = "bottom", ha = "left")

    u_str = @sprintf("\$U_{bulk} = (%.1f, %.1f, %.1f)\$ km s\$^{-1}\$\n\$T = %.2f\$ eV", d.u_bulk_kms[1], d.u_bulk_kms[2], d.u_bulk_kms[3], d.T_eV)
    ax_d.text(0.04, 0.93, u_str; transform = ax_d.transAxes, color = "white", fontsize = 13.5, va = "top", ha = "left", linespacing = 1.4)
    cax_d = ax_d.inset_axes([1.035, 0.0, 0.046, 1.0])
    cb_d = fig.colorbar(im_d; cax = cax_d)
    cb_d.ax.tick_params(labelsize = 13.5)
    cb_d.set_label("\$\\int f\\,dv_y\$ [cm\$^{-3}\$ (km s\$^{-1}\$)\$^{-2}\$]"; fontsize = 14.5)

    # (e) 1D PSD vs Energy step plot
    pos_e = d.psd_1d .> 0.0
    if any(pos_e)
        ax_e.step(d.ec[pos_e], d.psd_1d[pos_e]; where = "mid", color = "black", lw = 1.9)
    end
    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlim(4.0, 3.5e3)
    ax_e.set_ylim(1.0e-21, 1.0e-5)
    ax_e.set_yticks(10.0 .^ (-21:2:-5))
    ax_e.grid(true; which = "major", ls = "--", lw = 0.75, color = "0.72", alpha = 0.85)
    ax_e.grid(true; which = "minor", ls = ":", lw = 0.5, color = "0.85", alpha = 0.65)
    ax_e.tick_params(labelsize = 14)
    ax_e.set_xlabel("Energy (eV)"; fontsize = 16)
    ax_e.set_ylabel("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 16)
    ax_e.text(-0.18, 1.02, "(e)"; transform = ax_e.transAxes, fontsize = 20, fontweight = "bold", va = "bottom", ha = "left")

    # (f) 2D Energy vs Pitch angle PSD
    ax_f.set_facecolor("black")
    epa_plot = copy(d.epa_psd) # (n_E, n_PA) -> y is Energy, x is Pitch angle
    epa_plot[epa_plot .<= 0.0] .= NaN
    norm_f = matplotlib.colors.LogNorm(vmin = 5.0e-9, vmax = 3.0e-6)
    im_f = ax_f.pcolormesh(d.pa_edges, d.edges, epa_plot; cmap = "turbo", norm = norm_f, shading = "auto", rasterized = true)
    ax_f.set_yscale("log")
    ax_f.set_xlim(0.0, 180.0)
    ax_f.set_ylim(6.0, 3.0e3)
    ax_f.set_xticks(0:30:180)
    ax_f.tick_params(labelsize = 14)
    ax_f.set_xlabel("Pitch angle (deg)"; fontsize = 16)
    ax_f.set_ylabel("Energy (eV)"; fontsize = 16)
    ax_f.text(-0.18, 1.02, "(f)"; transform = ax_f.transAxes, fontsize = 20, fontweight = "bold", va = "bottom", ha = "left")

    b_str = @sprintf("\$B = (%.2f, %.2f, %.2f)\$ nT", d.b_nT[1], d.b_nT[2], d.b_nT[3])
    ax_f.text(0.04, 0.92, b_str; transform = ax_f.transAxes, color = "white", fontsize = 13.5, va = "top", ha = "left")
    cax_f = ax_f.inset_axes([1.035, 0.0, 0.046, 1.0])
    cb_f = fig.colorbar(im_f; cax = cax_f)
    cb_f.ax.tick_params(labelsize = 13.5)
    cb_f.set_label("PSD (s\$^3\$ m\$^{-6}\$)"; fontsize = 14.5)

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 3 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "perlmutter_runs", "mc_400km_without_crustal_cum_005_paper")
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig3.png")
    rdir = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig3(pdir, out; raw_run_dir = rdir)
end

