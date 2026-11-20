# Publication-quality script for Figure 1 of the Mars O2+ test-particle manuscript:
#   (a) Magnetic field |B| and unit vectors in the XY plane (Z = 0)
#   (b) Convective electric field |E_{C,xz}| and unit vectors in the XZ plane (Y = 0)
#   (c) Hall electric field |E_{H,xz}| and unit vectors in the XZ plane (Y = 0)
#   (d) Ambipolar electric field |E_{A,xz}| and unit vectors in the XZ plane (Y = 0)
#
# Usage:
#   julia --project=. examples/forward_tracing/monte_carlo_forward_tracing/plot_fig1.jl [OUTPUT_PNG] [DAT_OR_VTS_PATH]

using MarsTP
using Batsrus
using StaticArrays
using LinearAlgebra
using Printf
using PyPlot

if !isdefined(@__MODULE__, :paper_style)
    include(joinpath(@__DIR__, "plot_style.jl"))
end

const Vec3 = SVector{3, Float64}

function _default_fig1_field_path()
    dat_path = MarsTP.project_path("BATSRUS-data", "MHD-without-crustal", "3d__var_1_n00150000.dat")
    return isfile(dat_path) ? dat_path : MarsTP.data_path("mars_fields_without_crustal.vts")
end

function _resolve_fig1_field_path(vts_path::AbstractString)
    resolved = MarsTP.resolve_project_path(vts_path)
    endswith(lowercase(resolved), ".dat") && return resolved
    bname = lowercase(basename(resolved))
    if occursin("without_crustal", bname)
        dat_no = MarsTP.project_path("BATSRUS-data", "MHD-without-crustal", "3d__var_1_n00150000.dat")
        isfile(dat_no) && return dat_no
    elseif occursin("with_crustal", bname) || occursin("crustal", bname)
        dat_wc = MarsTP.project_path("BATSRUS-data", "MHD-with-crustal", "3d__var_1_n00120000.dat")
        isfile(dat_wc) && return dat_wc
    end
    return resolved
end

function plot_fig1(
        out_png::AbstractString = joinpath(pwd(), "outputs", "paper_figures_without_crustal", "fig1.png");
        vts_path::AbstractString = _default_fig1_field_path(),
    )
    resolved_path = _resolve_fig1_field_path(vts_path)
    if endswith(lowercase(resolved_path), ".dat")
        println("Loading native BATSRUS Tecplot file via Batsrus.jl: ", resolved_path)
        bat = load_batsrus_tecplot(resolved_path; polar_repair = true)
        r_ax, th_ax, ph_ax = bat.r, bat.theta, bat.phi
        B_sph = MarsTP._rotate_vectors_to_spherical!(copy(bat.B_T), r_ax, th_ax, ph_ax)
        Ec_sph = MarsTP._rotate_vectors_to_spherical!(copy(bat.E_conv), r_ax, th_ax, ph_ax)
        Eh_sph = MarsTP._rotate_vectors_to_spherical!(copy(bat.E_hall), r_ax, th_ax, ph_ax)
        Ea_sph = MarsTP._rotate_vectors_to_spherical!(copy(bat.E_gradpe), r_ax, th_ax, ph_ax)
        b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, B_sph, r_ax, th_ax, ph_ax)
        ec_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, Ec_sph, r_ax, th_ax, ph_ax)
        eh_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, Eh_sph, r_ax, th_ax, ph_ax)
        ea_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, Ea_sph, r_ax, th_ax, ph_ax)
        r_min_Rm = first(r_ax) / Rm
        r_max_m = last(r_ax) * (1.0 - 1.0e-7)
    else
        fields = load_mhd_fields(resolved_path)
        b_itp = MarsTP.TP.build_interpolator(MarsTP.TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
        work_itp = build_field_work_interpolators(resolved_path; include_extended = true)
        ec_itp = work_itp.conv
        eh_itp = work_itp.hall
        ea_itp = work_itp.gradpe
        r_min_Rm = Rinner / Rm
        r_max_m = maximum(fields.r) * (1.0 - 1.0e-7)
    end

    xs = collect(range(-4.0, 2.0; length = 301))
    ys = collect(range(-4.0, 4.0; length = 401))
    nx, ny = length(xs), length(ys)

    b_mag = fill(NaN, ny, nx)
    ec_xz = fill(NaN, ny, nx)
    eh_xz = fill(NaN, ny, nx)
    ea_xz = fill(NaN, ny, nx)

    clamp_r(pos::Vec3) = norm(pos) > r_max_m ? pos * (r_max_m / norm(pos)) : (norm(pos) < r_min_Rm * Rm ? pos * ((r_min_Rm * Rm * (1.0 + 1.0e-7)) / norm(pos)) : pos)

    for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
        r_xy = hypot(x, y)
        if r_xy >= r_min_Rm
            pos_xy = clamp_r(Vec3(x * Rm, y * Rm, 0.0))
            bv = Vec3(b_itp(pos_xy)) * 1.0e9 # nT
            b_mag[iy, ix] = max(norm(bv), 1.0e-12)

            pos_xz = clamp_r(Vec3(x * Rm, 0.0, y * Rm))
            ec = Vec3(ec_itp(pos_xz)) * 1.0e3 # mV/m
            eh = Vec3(eh_itp(pos_xz)) * 1.0e3 # mV/m
            ea = Vec3(ea_itp(pos_xz)) * 1.0e3 # mV/m
            ec_xz[iy, ix] = max(hypot(ec[1], ec[3]), 1.0e-12)
            eh_xz[iy, ix] = max(hypot(eh[1], eh[3]), 1.0e-12)
            ea_xz[iy, ix] = max(hypot(ea[1], ea[3]), 1.0e-12)
        end
    end

    # Coarser quiver grid so arrows can be larger and thicker
    xq = collect(-3.75:0.35:1.85)
    yq = collect(-3.85:0.35:3.85)
    nqx, nqy = length(xq), length(yq)
    bx_u = fill(NaN, nqy, nqx); by_u = fill(NaN, nqy, nqx)
    ecx_u = fill(NaN, nqy, nqx); ecz_u = fill(NaN, nqy, nqx)
    ehx_u = fill(NaN, nqy, nqx); ehz_u = fill(NaN, nqy, nqx)
    eax_u = fill(NaN, nqy, nqx); eaz_u = fill(NaN, nqy, nqx)

    for (ix, x) in enumerate(xq), (iy, y) in enumerate(yq)
        hypot(x, y) < 1.15 && continue
        bv = Vec3(b_itp(clamp_r(Vec3(x * Rm, y * Rm, 0.0))))
        bn = hypot(bv[1], bv[2])
        if bn > 0
            bx_u[iy, ix] = bv[1] / bn
            by_u[iy, ix] = bv[2] / bn
        end

        pos_xz = clamp_r(Vec3(x * Rm, 0.0, y * Rm))
        ec = Vec3(ec_itp(pos_xz))
        ecn = hypot(ec[1], ec[3])
        if ecn > 0
            ecx_u[iy, ix] = ec[1] / ecn
            ecz_u[iy, ix] = ec[3] / ecn
        end

        eh = Vec3(eh_itp(pos_xz))
        ehn = hypot(eh[1], eh[3])
        if ehn > 0
            ehx_u[iy, ix] = eh[1] / ehn
            ehz_u[iy, ix] = eh[3] / ehn
        end

        ea = Vec3(ea_itp(pos_xz))
        ean = hypot(ea[1], ea[3])
        if ean > 0
            eax_u[iy, ix] = ea[1] / ean
            eaz_u[iy, ix] = ea[3] / ean
        end
    end

    st = paper_style(13.2; font_scale = 1.15)

    fig, axes = plt.subplots(2, 2; figsize = (13.2, 12.8), layout = "constrained")
    fig.set_constrained_layout_pads(w_pad = 0.02, h_pad = 0.02, wspace = 0.04, hspace = 0.03)

    cmap_c = paper_cmap_continuous()
    panels = (
        (axes[1, 1], b_mag, bx_u, by_u, "(a)", "Magnetic field, XY", "Y (Rm)", L"$|\mathbf{B}|$ (nT)", (1.0e0, 4.0e1)),
        (axes[1, 2], ec_xz, ecx_u, ecz_u, "(b)", "Convective electric field, XZ", "Z (Rm)", L"$|\mathbf{E}_{C,xz}|$ (mV/m)", (1.0e-2, 4.0e0)),
        (axes[2, 1], eh_xz, ehx_u, ehz_u, "(c)", "Hall electric field, XZ", "Z (Rm)", L"$|\mathbf{E}_{H,xz}|$ (mV/m)", (1.0e-5, 1.0e0)),
        (axes[2, 2], ea_xz, eax_u, eaz_u, "(d)", "Ambipolar electric field, XZ", "Z (Rm)", L"$|\mathbf{E}_{A,xz}|$ (mV/m)", (2.0e-5, 3.0e-1)),
    )

    th = range(0.0, 2pi; length = 240)
    for (ax, mat, ux, uy, tag, title_str, ylabel_str, cbar_lbl, (vmin, vmax)) in panels
        norm_c = matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)
        im = ax.pcolormesh(xs, ys, mat; cmap = cmap_c, norm = norm_c, shading = "auto", rasterized = true)
        ax.quiver(
            xq, yq, ux, uy;
            color = "black", pivot = "mid", scale = 20.5, width = 0.0046,
            headwidth = 4.2, headlength = 5.0, headaxislength = 4.4, zorder = 3,
        )
        plot_mars(ax; radius = 1.0, zorder = 5)
        ax.plot(cos.(th), sin.(th); color = REF_ON_DARK, lw = 1.8, zorder = 6)
        ax.plot(r_min_Rm .* cos.(th), r_min_Rm .* sin.(th); color = "0.55", lw = 1.1, zorder = 6)

        ax.set_xlim(-4.0, 2.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_aspect("equal")
        ax.set_xticks([-4, -2, 0, 2])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.tick_params(axis = "both", which = "major", labelsize = st.tick, length = st.maj_sz, width = st.maj_w)
        ax.set_xlabel("X (Rm)"; fontsize = st.label)
        ax.set_ylabel(ylabel_str; fontsize = st.label)
        ax.set_title(title_str; fontsize = st.title, pad = 9)
        ax.text(
            0.03, 0.95, tag;
            transform = ax.transAxes, fontsize = st.panel, fontweight = "bold",
            va = "top", ha = "left",
            bbox = Dict("boxstyle" => "square,pad=0.2", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.9),
            zorder = 10,
        )
        cb = fig.colorbar(im; ax = ax, fraction = 0.056, pad = 0.03)
        style_colorbar!(cb, st; label = cbar_lbl)
    end

    mkpath(dirname(abspath(out_png)))
    fig.savefig(out_png; dpi = 300, bbox_inches = "tight")
    plt.close(fig)
    println("Saved Figure 1 to ", abspath(out_png))
    return out_png
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "outputs", "paper_figures_without_crustal", "fig1.png")
    vts = length(ARGS) >= 2 ? ARGS[2] : _default_fig1_field_path()
    plot_fig1(out; vts_path = vts)
end
