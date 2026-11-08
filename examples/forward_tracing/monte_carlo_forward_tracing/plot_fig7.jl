# Publication-quality script for Figure 7 of the Mars O2+ test-particle manuscript:
#   3 rows x 2 columns for representative tail ions at:
#     Column 1: Location at (-2, 0, +0.75) R_M (current sheet center)
#     Column 2: Location at (-2, -0.5, +0.75) R_M (magnetotail lobe)
#     (a, b) XZ trajectory projection with Mars disk, numbered markers 1 & 2, and E_C, E_H, E_A vectors
#     (c, d) Instantaneous power qE·v (W) vs time since release (s) on symlog scale
#     (e, f) Cumulative work ε (eV) vs time since release (s) on symlog scale with right-side summary
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig7.jl [PAPER_DIR] [OUTPUT_PNG] [VTS_PATH]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :_draw_field_vectors!)
    include(joinpath(@__DIR__, "plot_fig5.jl"))
end

function plot_fig7(
        paper_dir::AbstractString = _default_paper_dir(),
        out_png::AbstractString = joinpath(paper_dir, "fig7.png");
        vts_path::Union{Nothing, AbstractString} = nothing,
        use_x175::Bool = false,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    rep, rm_m, fields_file = jldopen(jld_path, "r") do f
        key = (!use_x175 && haskey(f, "fig7_rep_20")) ? "fig7_rep_20" : (haskey(f, "fig7_rep_175") ? "fig7_rep_175" : nothing)
        key === nothing && error("Representative trajectories `fig7_rep_20` not found in $jld_path; re-run analyze_paper_results.jl")
        r_m = haskey(f, "rm_m") ? Float64(f["rm_m"]) : Rm
        ff = haskey(f, "fields_file") ? String(f["fields_file"]) : "mars_fields_without_crustal.vts"
        (f[key], r_m, ff)
    end

    resolved_vts = if vts_path !== nothing && isfile(vts_path)
        String(vts_path)
    elseif isfile(MarsTP.data_path(fields_file))
        MarsTP.data_path(fields_file)
    else
        MarsTP.data_path("mars_fields_without_crustal.vts")
    end
    work_itp = build_field_work_interpolators(resolved_vts; include_extended = true)
    qe = MarsTP.TP.qᵢ

    PyPlot.matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    PyPlot.matplotlib.rcParams["font.size"] = 18
    PyPlot.matplotlib.rcParams["xtick.labelsize"] = 18.5
    PyPlot.matplotlib.rcParams["ytick.labelsize"] = 18.5
    PyPlot.matplotlib.rcParams["xtick.major.size"] = 7.5
    PyPlot.matplotlib.rcParams["ytick.major.size"] = 7.5
    PyPlot.matplotlib.rcParams["xtick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["ytick.major.width"] = 1.4
    PyPlot.matplotlib.rcParams["xtick.minor.size"] = 4.5
    PyPlot.matplotlib.rcParams["ytick.minor.size"] = 4.5
    PyPlot.matplotlib.rcParams["xtick.minor.width"] = 1.1
    PyPlot.matplotlib.rcParams["ytick.minor.width"] = 1.1

    fig, axes = plt.subplots(3, 2; figsize = (15.2, 11.2), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.04, h_pad = 0.03, wspace = 0.06, hspace = 0.03)

    col_configs = (
        (1, "Location at (\$−2, 0, +0.75\$) \$R_M\$", ("a", "c", "e"), (-2.5, 0.5), (0.0, 2.0), (0.38, 0.90)),
        (2, "Location at (\$−2, −0.5, +0.75\$) \$R_M\$", ("b", "d", "f"), (-2.5, 1.1), (-0.7, 1.7), (0.58, 0.83)),
    )

    for (col, title_str, (tag_tr, tag_p, tag_w), xlim_tr, ylim_tr, frac_markers) in col_configs
        t_vec = rep.t[col]
        u_mat = rep.u[col]
        isempty(t_vec) && continue
        u_sv = [SVector{6, Float64}(u_mat[:, k]) for k in 1:size(u_mat, 2)]
        tr = (; t = t_vec, u = u_sv)
        prof = Electric_field_work_profile(tr, work_itp; species = "O2+")

        xs = [u[1] / rm_m for u in u_sv]
        zs = [u[3] / rm_m for u in u_sv]
        t_rel = prof.t .- prof.t[1]
        t_end = t_rel[end]

        m_idxs = [clamp(searchsortedlast(t_rel, f * t_end), 2, length(t_rel) - 1) for f in frac_markers]

        # Row 1: XZ trajectory + Mars disk + E vectors
        ax_tr = axes[1, col]
        plot_mars(ax_tr; radius = 1.0, zorder = 2)
        ax_tr.plot(xs, zs; color = "black", ls = "--", lw = 1.7, zorder = 5)
        if length(xs) >= 3
            ax_tr.annotate(
                "", xy = (xs[end], zs[end]), xytext = (xs[end - 2], zs[end - 2]),
                arrowprops = Dict("arrowstyle" => "->", "color" => "black", "lw" => 1.8, "mutation_scale" => 15),
                zorder = 6,
            )
        end
        for (m_num, idx) in enumerate(m_idxs)
            pos_m = SVector{3, Float64}(u_sv[idx][1], u_sv[idx][2], u_sv[idx][3])
            _draw_field_vectors!(ax_tr, pos_m, work_itp, rm_m; arrow_len = 0.34)
            _draw_circled_number!(ax_tr, xs[idx], zs[idx], m_num; zorder = 10)
        end
        if col == 1
            for (lbl, c_hex) in (("\$E_C\$", _COL_EC), ("\$E_H\$", _COL_EH), ("\$E_A\$", _COL_EA))
                ax_tr.plot(Float64[], Float64[]; color = c_hex, lw = 2.4, label = lbl)
            end
            ax_tr.legend(loc = "center left", bbox_to_anchor = (0.03, 0.72), frameon = false, fontsize = 17)
        end
        ax_tr.set_xlim(xlim_tr)
        ax_tr.set_ylim(ylim_tr)
        ax_tr.set_yticks(-0.5:0.5:2.0)
        ax_tr.set_ylim(ylim_tr)
        ax_tr.set_aspect("equal")
        ax_tr.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
        ax_tr.set_xlabel("\$X_{MSE}\$ (\$R_M\$)"; fontsize = 19.5)
        ax_tr.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = 19.5)
        ax_tr.set_title(title_str; fontsize = 20.5, fontweight = "bold", pad = 9)
        ax_tr.text(0.03, 0.94, tag_tr; transform = ax_tr.transAxes, fontsize = 22, fontweight = "bold", va = "top", ha = "left")

        # Row 2: Instantaneous power qE·v (W) on symlog scale
        ax_p = axes[2, col]
        ax_p.axhline(0.0; color = "0.75", lw = 0.8)
        ax_p.plot(t_rel, prof.power_conv_eV_s .* qe; color = _COL_EC, lw = 1.8, label = "\$P_C\$")
        ax_p.plot(t_rel, prof.power_hall_eV_s .* qe; color = _COL_EH, lw = 1.8, label = "\$P_H\$")
        ax_p.plot(t_rel, prof.power_gradpe_eV_s .* qe; color = _COL_EA, lw = 1.8, label = "\$P_A\$")
        ax_p.plot(t_rel, prof.power_total_eV_s .* qe; color = "black", ls = "--", lw = 1.8, label = "\$P_T\$")
        ax_p.set_yscale("symlog"; linthresh = 1.0e-19, linscale = 0.8)
        ax_p.set_xlim(0.0, t_end * 1.04)
        ax_p.tick_params(axis = "both", which = "major", labelbottom = false, labelsize = 18.5, length = 7.5, width = 1.4)
        ax_p.tick_params(axis = "y", which = "minor", length = 4.5, width = 1.1)
        for (m_num, idx) in enumerate(m_idxs)
            tm = t_rel[idx]
            ax_p.axvline(tm; color = "0.65", ls = ":", lw = 1.1, zorder = 1)
            _draw_circled_number!(ax_p, tm, 0.10, m_num; transform = ax_p.get_xaxis_transform(), zorder = 8)
        end
        if col == 1
            ax_p.legend(loc = "upper left", bbox_to_anchor = (1.02, 0.98), frameon = false, fontsize = 17)
        end
        ax_p.set_ylabel("\$q\\mathbf{E} \\cdot \\mathbf{v}\$ (W)"; fontsize = 19.5)
        ax_p.text(0.03, 0.94, tag_p; transform = ax_tr === nothing ? ax_p.transAxes : ax_p.transAxes, fontsize = 22, fontweight = "bold", va = "top", ha = "left")

        # Row 3: Cumulative work ε (eV) on symlog scale + right-side text summary
        ax_w = axes[3, col]
        ax_w.axhline(0.0; color = "0.75", lw = 0.8)
        ax_w.plot(t_rel, prof.conv_eV; color = _COL_EC, lw = 1.8)
        ax_w.plot(t_rel, prof.hall_eV; color = _COL_EH, lw = 1.8)
        ax_w.plot(t_rel, prof.gradpe_eV; color = _COL_EA, lw = 1.8)
        ax_w.plot(t_rel, prof.total_eV; color = "black", ls = "--", lw = 1.8)
        ax_w.set_yscale("symlog"; linthresh = 1.0, linscale = 0.6)
        ax_w.set_xlim(0.0, t_end * 1.04)
        ax_w.tick_params(axis = "both", which = "major", labelsize = 18.5, length = 7.5, width = 1.4)
        ax_w.tick_params(axis = "y", which = "minor", length = 4.5, width = 1.1)
        ax_w.set_xlabel("Time since release (s)"; fontsize = 19.5)
        ax_w.set_ylabel("\$\\varepsilon\$ (eV)"; fontsize = 19.5)
        ax_w.text(0.03, 0.94, tag_w; transform = ax_w.transAxes, fontsize = 22, fontweight = "bold", va = "top", ha = "left")

        wc_f = prof.conv_eV[end]
        wh_f = prof.hall_eV[end]
        wa_f = prof.gradpe_eV[end]
        wt_f = prof.total_eV[end]
        denom = abs(wt_f) > 1.0e-9 ? wt_f : 1.0
        ax_w.text(1.03, 0.84, @sprintf("\$\\varepsilon_C = %.0f\$ eV\n(%.1f%%)", wc_f, 100.0 * wc_f / denom); transform = ax_w.transAxes, color = _COL_EC, fontsize = 17, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.03, 0.57, @sprintf("\$\\varepsilon_H = %.0f\$ eV\n(%.1f%%)", wh_f, 100.0 * wh_f / denom); transform = ax_w.transAxes, color = _COL_EH, fontsize = 17, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.03, 0.30, @sprintf("\$\\varepsilon_A = %.0f\$ eV\n(%.1f%%)", wa_f, 100.0 * wa_f / denom); transform = ax_w.transAxes, color = _COL_EA, fontsize = 17, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.03, 0.07, @sprintf("\$\\varepsilon_T = %.0f\$ eV", wt_f); transform = ax_w.transAxes, color = "black", fontsize = 17, va = "center", ha = "left")
    end

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 7 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig7.png")
    vts = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig7(pdir, out; vts_path = vts)
end

