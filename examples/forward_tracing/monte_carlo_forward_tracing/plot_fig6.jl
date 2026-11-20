# Publication-quality script for Figure 6 of the Mars O2+ test-particle manuscript:
#   3 rows (for orbit angles 90°, 120°, 150° at r = 2.5 Rm) x 3 columns:
#     (a, b, c) XZ trajectory projection with Mars disk, numbered markers, and E_C, E_H, E_A field vectors
#     (d, e, f) Instantaneous power qE·v (W) vs time since release (s) on symlog scale
#     (g, h, i) Cumulative work ε (eV) vs time since release (s) on symlog scale with right-side summary
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig6.jl [PAPER_DIR] [OUTPUT_PNG] [VTS_PATH]

using MarsTP
using JLD2
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :paper_style)
    include(joinpath(@__DIR__, "plot_style.jl"))
end

const _COL_EC = COL_CONV
const _COL_EH = COL_HALL
const _COL_EA = COL_AMB

function _draw_field_vectors!(ax, pos_m::SVector{3, Float64}, work_itp, rm_m::Float64; arrow_len::Float64 = 0.36)
    x_Rm = pos_m[1] / rm_m
    z_Rm = pos_m[3] / rm_m
    for (itp_fn, col) in ((work_itp.conv, _COL_EC), (work_itp.hall, _COL_EH), (work_itp.gradpe, _COL_EA))
        ev = SVector{3, Float64}(itp_fn(pos_m))
        mag_xz = hypot(ev[1], ev[3])
        if mag_xz > 0
            dx = arrow_len * (ev[1] / mag_xz)
            dz = arrow_len * (ev[3] / mag_xz)
            ax.annotate(
                "", xy = (x_Rm + dx, z_Rm + dz), xytext = (x_Rm, z_Rm),
                arrowprops = Dict("arrowstyle" => "-|>", "color" => col, "lw" => 1.5, "mutation_scale" => 11),
                zorder = 8,
            )
        end
    end
    return nothing
end

function _draw_circled_number!(ax, x::Real, y::Real, num::Int; transform = nothing, zorder::Int = 10, fontsize::Real = 14.5)
    kw = Dict{Symbol, Any}(
        :fontsize => fontsize,
        :fontweight => "bold",
        :ha => "center",
        :va => "center",
        :bbox => Dict("boxstyle" => "circle,pad=0.18", "facecolor" => "white", "edgecolor" => "0.40", "lw" => 0.95),
        :zorder => zorder,
    )
    transform !== nothing && (kw[:transform] = transform)
    return ax.text(x, y, string(num); kw...)
end

function _default_paper_dir_fig6()
    p1 = joinpath(pwd(), "outputs", "paper_figures_without_crustal")
    isfile(joinpath(p1, "paper_analysis.jld2")) && return p1
    return joinpath(pwd(), "outputs", "paper_figures_with_crustal")
end

function plot_fig6(
        paper_dir::AbstractString = _default_paper_dir_fig6(),
        out_png::AbstractString = joinpath(paper_dir, "fig6.png");
        vts_path::Union{Nothing, AbstractString} = nothing,
        use_r20::Bool = false,
    )
    jld_path = joinpath(paper_dir, "paper_analysis.jld2")
    isfile(jld_path) || error("paper_analysis.jld2 not found in $paper_dir")

    rep, rm_m, fields_file = jldopen(jld_path, "r") do f
        key = (!use_r20 && haskey(f, "fig5_rep_25")) ? "fig5_rep_25" : (haskey(f, "fig5_rep_20") ? "fig5_rep_20" : nothing)
        key === nothing && error("Representative trajectories `fig5_rep_25` not found in $jld_path; re-run analyze_paper_results.jl")
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

    st = paper_style(15.8)

    fig, axes = plt.subplots(3, 3; figsize = (15.8, 12.0), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.03, h_pad = 0.03, wspace = 0.03, hspace = 0.03)

    row_configs = (
        (1, 90, ("(a)", "(d)", "(g)"), (-1.1, 1.7), (0.35, 0.82, 0.93)),
        (2, 120, ("(b)", "(e)", "(h)"), (-1.9, 1.05), (0.16, 0.92)),
        (3, 150, ("(c)", "(f)", "(i)"), (-2.4, 0.55), (0.2, 0.78)),
    )

    for (row, ang_deg, (tag_tr, tag_p, tag_w), xlim_tr, frac_markers) in row_configs
        t_vec = rep.t[row]
        u_mat = rep.u[row]
        isempty(t_vec) && continue
        u_sv = [SVector{6, Float64}(u_mat[:, k]) for k in 1:size(u_mat, 2)]
        tr = (; t = t_vec, u = u_sv)
        prof = Electric_field_work_profile(tr, work_itp; species = "O2+")

        xs = [u[1] / rm_m for u in u_sv]
        zs = [u[3] / rm_m for u in u_sv]
        t_rel = prof.t .- prof.t[1]
        t_end = t_rel[end]

        # Marker indices along time (clamped to points inside the visible XZ plot box)
        vis_idxs = findall(i -> (xlim_tr[1] + 0.12 <= xs[i] <= xlim_tr[2] - 0.12) && (0.62 <= zs[i] <= 2.85), eachindex(t_rel))
        m_idxs = map(frac_markers) do f
            idx0 = clamp(searchsortedlast(t_rel, f * t_end), 2, length(t_rel) - 1)
            if isempty(vis_idxs) || (idx0 in vis_idxs)
                return idx0
            end
            _, best_pos = findmin(i -> abs(t_rel[i] - f * t_end), vis_idxs)
            return vis_idxs[best_pos]
        end

        # Column 1: XZ trajectory + Mars disk + E vectors
        ax_tr = axes[row, 1]
        plot_mars(ax_tr; radius = 1.0, zorder = 2)
        ax_tr.plot(xs, zs; color = REF_BLACK, ls = "--", lw = 1.8, zorder = 5)
        if length(xs) >= 3
            ax_tr.annotate(
                "", xy = (xs[end], zs[end]), xytext = (xs[end - 2], zs[end - 2]),
                arrowprops = Dict("arrowstyle" => "->", "color" => REF_BLACK, "lw" => 1.8, "mutation_scale" => 15),
                zorder = 6,
            )
        end
        for (m_num, idx) in enumerate(m_idxs)
            pos_m = SVector{3, Float64}(u_sv[idx][1], u_sv[idx][2], u_sv[idx][3])
            _draw_field_vectors!(ax_tr, pos_m, work_itp, rm_m; arrow_len = 0.36)
            _draw_circled_number!(ax_tr, xs[idx], zs[idx], m_num; zorder = 10, fontsize = st.annot_sm)
        end
        if row == 1
            for (lbl, col) in (("\$E_C\$", _COL_EC), ("\$E_H\$", _COL_EH), ("\$E_A\$", _COL_EA))
                ax_tr.plot(Float64[], Float64[]; color = col, lw = 2.6, label = lbl)
            end
            ax_tr.legend(loc = "center left", bbox_to_anchor = (0.02, 0.56), frameon = false, fontsize = st.legend)
        end
        ax_tr.set_xlim(xlim_tr)
        ax_tr.set_ylim(0.5, 3.0)
        ax_tr.set_aspect("equal")
        ax_tr.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        ax_tr.set_ylabel("\$Z_{MSE}\$ (\$R_M\$)"; fontsize = st.label)
        row == 3 && ax_tr.set_xlabel("\$X_{MSE}\$ (\$R_M\$)"; fontsize = st.label)
        ax_tr.text(0.03, 0.94, tag_tr; transform = ax_tr.transAxes, fontsize = st.panel, fontweight = "bold", va = "top", ha = "left")
        ax_tr.text(0.96, 0.94, "$(ang_deg)°"; transform = ax_tr.transAxes, fontsize = st.title, fontweight = "bold", va = "top", ha = "right")

        # Column 2: Instantaneous power qE·v (W) on symlog scale
        ax_p = axes[row, 2]
        ax_p.axhline(0.0; color = REF_GRAY, lw = 0.9)
        ax_p.plot(t_rel, prof.power_conv_eV_s .* qe; color = _COL_EC, lw = st.lw_curve, label = "\$P_C\$")
        ax_p.plot(t_rel, prof.power_hall_eV_s .* qe; color = _COL_EH, lw = st.lw_curve, label = "\$P_H\$")
        ax_p.plot(t_rel, prof.power_gradpe_eV_s .* qe; color = _COL_EA, lw = st.lw_curve, label = "\$P_A\$")
        ax_p.plot(t_rel, prof.power_total_eV_s .* qe; color = COL_TOTAL, ls = "--", lw = st.lw_curve, label = "\$P_T\$")
        ax_p.set_yscale("symlog"; linthresh = 1.0e-19, linscale = 0.6)
        ax_p.set_xlim(0.0, t_end * 1.04)
        for (m_num, idx) in enumerate(m_idxs)
            tm = t_rel[idx]
            ax_p.axvline(tm; color = REF_GRAY, ls = ":", lw = 1.1, zorder = 1)
            _draw_circled_number!(ax_p, tm, 0.12, m_num; transform = ax_p.get_xaxis_transform(), zorder = 8, fontsize = st.annot_sm)
        end
        if row == 1
            ax_p.legend(loc = "upper center", ncol = 2, frameon = false, fontsize = st.legend, columnspacing = 0.9, handletextpad = 0.4)
        end
        ax_p.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        ax_p.tick_params(axis = "y", which = "minor", length = st.min_sz, width = st.min_w)
        ax_p.grid(true; which = "major", ls = "--", lw = 0.7, color = "0.78", alpha = 0.75)
        ax_p.set_ylabel("\$q\\mathbf{E} \\cdot \\mathbf{v}\$ (W)"; fontsize = st.label)
        row == 3 && ax_p.set_xlabel("Time since release (s)"; fontsize = st.label)
        ax_p.text(0.03, 0.94, tag_p; transform = ax_p.transAxes, fontsize = st.panel, fontweight = "bold", va = "top", ha = "left")

        # Column 3: Cumulative work ε (eV) on symlog scale + right-side text summary
        ax_w = axes[row, 3]
        ax_w.axhline(0.0; color = REF_GRAY, lw = 0.9)
        ax_w.plot(t_rel, prof.conv_eV; color = _COL_EC, lw = st.lw_curve)
        ax_w.plot(t_rel, prof.hall_eV; color = _COL_EH, lw = st.lw_curve)
        ax_w.plot(t_rel, prof.gradpe_eV; color = _COL_EA, lw = st.lw_curve)
        ax_w.plot(t_rel, prof.total_eV; color = COL_TOTAL, ls = "--", lw = st.lw_curve)
        ax_w.set_yscale("symlog"; linthresh = 1.0, linscale = 0.6)
        ax_w.set_xlim(0.0, t_end * 1.04)
        ax_w.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        ax_w.tick_params(axis = "y", which = "minor", length = st.min_sz, width = st.min_w)
        ax_w.grid(true; which = "major", ls = "--", lw = 0.7, color = "0.78", alpha = 0.75)
        ax_w.set_ylabel("\$\\varepsilon\$ (eV)"; fontsize = st.label)
        row == 3 && ax_w.set_xlabel("Time since release (s)"; fontsize = st.label)
        ax_w.text(0.03, 0.94, tag_w; transform = ax_w.transAxes, fontsize = st.panel, fontweight = "bold", va = "top", ha = "left")

        wc_f = prof.conv_eV[end]
        wh_f = prof.hall_eV[end]
        wa_f = prof.gradpe_eV[end]
        wt_f = prof.total_eV[end]
        denom = abs(wt_f) > 1.0e-9 ? wt_f : 1.0
        ax_w.text(1.04, 0.84, @sprintf("\$\\varepsilon_C = %.0f\$ eV\n(%.1f%%)", wc_f, 100.0 * wc_f / denom); transform = ax_w.transAxes, color = _COL_EC, fontweight = "bold", fontsize = st.annot, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.04, 0.57, @sprintf("\$\\varepsilon_H = %.0f\$ eV\n(%.1f%%)", wh_f, 100.0 * wh_f / denom); transform = ax_w.transAxes, color = _COL_EH, fontweight = "bold", fontsize = st.annot, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.04, 0.3, @sprintf("\$\\varepsilon_A = %.0f\$ eV\n(%.1f%%)", wa_f, 100.0 * wa_f / denom); transform = ax_w.transAxes, color = _COL_EA, fontweight = "bold", fontsize = st.annot, va = "center", ha = "left", linespacing = 1.15)
        ax_w.text(1.04, 0.07, @sprintf("\$\\varepsilon_T = %.0f\$ eV", wt_f); transform = ax_w.transAxes, color = COL_TOTAL, fontweight = "bold", fontsize = st.annot, va = "center", ha = "left")
    end

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 6 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    pdir = length(ARGS) >= 1 ? ARGS[1] : _default_paper_dir_fig6()
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(pdir, "fig6.png")
    vts = length(ARGS) >= 3 ? ARGS[3] : nothing
    plot_fig6(pdir, out; vts_path = vts)
end
