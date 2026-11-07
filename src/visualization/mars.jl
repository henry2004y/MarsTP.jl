# Mars disk/sphere and axisymmetric empirical BS/MPB, coordinates in Rm.

const MODELS = Dict{String, Vector{NTuple{3, Float64}}}(
    "BS" => [(0.6, 2.081, 1.026)],
    "MPB" => [(0.640, 1.080, 0.770), (1.600, 0.528, 1.009)],
)

"""
    boundary_profile(name, xmin=-5.0, samples=500; xmin=xmin, samples=samples)

Return separate `(x, rho)` branches in `Rm`. Exclude unphysical `r <= 0`.
"""
function boundary_profile(
        name::AbstractString,
        xmin_pos::Real = -5.0,
        samples_pos::Integer = 500;
        xmin::Real = xmin_pos,
        samples::Integer = samples_pos,
    )
    key = uppercase(String(name))
    if !haskey(MODELS, key) || samples < 3 || !isfinite(xmin)
        throw(ArgumentError("Expected BS/MPB, finite xmin and samples >= 3"))
    end
    branches = Tuple{Vector{Float64}, Vector{Float64}}[]
    theta = collect(range(0.0, pi, length = Int(samples)))
    for (k, (x0, length_par, eccentricity)) in enumerate(MODELS[key])
        denom = 1.0 .+ eccentricity .* cos.(theta)
        valid = denom .> 0.0
        r = length_par ./ denom[valid]
        th_v = theta[valid]
        x = x0 .+ r .* cos.(th_v)
        rho = r .* sin.(th_v)
        keep = x .>= Float64(xmin)
        if key == "MPB"
            keep .&= (k == 1 ? (x .>= 0.0) : (x .< 0.0))
        end
        push!(branches, (x[keep], rho[keep]))
    end
    return branches
end

function _is_3d_axis(ax)
    return try
        String(ax.name) == "3d"
    catch
        false
    end
end

"""
    plot_mars(ax; radius=1.0, center=nothing, color="#b97956", alpha=1.0,
              texture=nothing, texture_path=nothing, facecolor=nothing,
              edgecolor="#654b3c", lw=0.6, zorder=5)

Draw a bundled Mars disk in 2D, or a solid sphere in 3D using PyPlot/Matplotlib.
"""
function plot_mars(
        ax;
        radius::Real = 1.0,
        center = nothing,
        color = "#b97956",
        alpha::Real = 1.0,
        texture = nothing,
        texture_path = nothing,
        facecolor = nothing,
        edgecolor = "#654b3c",
        lw::Real = 0.6,
        zorder::Integer = 5,
    )
    isfinite(radius) && radius > 0 || throw(ArgumentError("radius must be positive"))
    r = Float64(radius)
    col = facecolor === nothing ? color : facecolor
    is3d = _is_3d_axis(ax)
    use_texture = texture === nothing ? !is3d : Bool(texture)
    if use_texture
        is3d && throw(ArgumentError("Disk images require a 2D axis; use texture=false for a sphere"))
        tex_file = texture_path === nothing ? joinpath(@__DIR__, "mars_globe_true_color.png") : String(texture_path)
        cx, cy = center === nothing ? (0.0, 0.0) : (Float64(center[1]), Float64(center[2]))
        img = PyPlot.plt.imread(tex_file)
        return ax.imshow(
            img;
            extent = (cx - r, cx + r, cy - r, cy + r),
            alpha = Float64(alpha),
            zorder = Int(zorder),
            origin = "upper",
        )
    end
    if is3d
        c = center === nothing ? (0.0, 0.0, 0.0) : (Float64(center[1]), Float64(center[2]), Float64(center[3]))
        u = range(0.0, 2pi, length = 60)
        v = range(0.0, pi, length = 30)
        xs = [c[1] + r * cos(ui) * sin(vi) for vi in v, ui in u]
        ys = [c[2] + r * sin(ui) * sin(vi) for vi in v, ui in u]
        zs = [c[3] + r * cos(vi) for vi in v, ui in u]
        return ax.plot_surface(
            xs, ys, zs;
            color = col,
            alpha = Float64(alpha),
            linewidth = 0,
            shade = true,
        )
    end
    c2 = center === nothing ? (0.0, 0.0) : (Float64(center[1]), Float64(center[2]))
    circle = PyPlot.matplotlib.patches.Circle(
        c2, r;
        facecolor = col,
        edgecolor = edgecolor,
        lw = Float64(lw),
        alpha = Float64(alpha),
        zorder = Int(zorder),
    )
    return ax.add_patch(circle)
end

"""
    bs_mpb(ax, plane="XZ"; draw_bs=true, draw_mpb=true, xmin=-5.0,
           x_slice=nothing, color="black", linewidth=0.8, sphere=false,
           boundary_color=nothing, boundary_ls=nothing, boundary_lw=nothing,
           mars_lw=0, mars_ls="-")

Draw bow shock (BS) and magnetic pileup boundary (MPB) conics in XY/XZ or surfaces in 3D;
YZ requires an `x_slice` in `Rm`.
"""
function bs_mpb(
        ax,
        plane_pos::AbstractString = "XZ";
        plane::AbstractString = plane_pos,
        draw_bs::Bool = true,
        draw_mpb::Bool = true,
        xmin::Real = -5.0,
        x_slice = nothing,
        color = "black",
        linewidth::Real = 0.8,
        sphere::Bool = false,
        boundary_color = nothing,
        boundary_ls = nothing,
        boundary_lw = nothing,
        mars_lw::Real = 0,
        mars_ls::AbstractString = "-",
    )
    p = uppercase(String(plane))
    p in ("XY", "XZ", "YZ", "3D") || throw(ArgumentError("plane must be XY, XZ, YZ or 3D"))
    col = boundary_color === nothing ? color : boundary_color
    lw = Float64(boundary_lw === nothing ? linewidth : boundary_lw)
    sphere && plot_mars(ax)
    artists = Any[]
    for (name, enabled, default_ls) in (("BS", draw_bs, "--"), ("MPB", draw_mpb, ":"))
        ls = boundary_ls === nothing ? default_ls : boundary_ls
        enabled || continue
        if p == "YZ"
            x_slice === nothing && continue
            isfinite(x_slice) || throw(ArgumentError("x_slice must be finite"))
            xs = Float64(x_slice)
            k = (name == "BS" || xs >= 0.0) ? 1 : 2
            x0, length_par, ecc = MODELS[name][k]
            r = length_par - ecc * (xs - x0)
            rho2 = r * r - (xs - x0)^2
            if r > 0.0 && rho2 >= 0.0
                circle = PyPlot.matplotlib.patches.Circle(
                    (0.0, 0.0), sqrt(rho2);
                    fill = false,
                    edgecolor = col,
                    ls = ls,
                    lw = lw,
                    label = name,
                )
                push!(artists, ax.add_patch(circle))
            end
            continue
        end
        for (x, rho) in boundary_profile(name; xmin = xmin)
            if p == "3D"
                phi = collect(range(0.0, 2pi, length = 45))
                xx = repeat(x, 1, length(phi))
                yy = rho .* cos.(phi)'
                zz = rho .* sin.(phi)'
                push!(artists, ax.plot_wireframe(
                    xx, yy, zz;
                    rcount = 16,
                    ccount = 12,
                    color = col,
                    linewidth = lw * 0.5,
                    alpha = 0.25,
                ))
            else
                for sgn in (1.0, -1.0)
                    append!(artists, ax.plot(x, sgn .* rho; color = col, ls = ls, lw = lw))
                end
            end
        end
    end
    return artists
end

"""
    plot_mars_context(ax, plane="XZ"; boundaries=true, kwargs...)

Common background used by trajectory projections.
"""
function plot_mars_context(
        ax,
        plane_pos::AbstractString = "XZ";
        plane::AbstractString = plane_pos,
        boundaries::Bool = true,
        kwargs...,
    )
    if boundaries
        bs_mpb(ax, plane; kwargs...)
    end
    plot_mars(ax)
    return ax
end

