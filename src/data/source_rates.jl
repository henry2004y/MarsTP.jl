struct O2plusSourceRates
    path::String
    r::Vector{Float64}
    theta::Vector{Float64}
    phi::Vector{Float64}
    production_density::Array{Float64, 3}
    production_rate::Array{Float64, 3}
    outflow_rate::Array{Float64, 2}
    cell_volume::Array{Float64, 3}
    surface_area::Array{Float64, 2}
    metadata::Dict{String, Any}
end

function _mat_value(data, name)
    haskey(data, name) || error("Missing variable $name")
    return data[name]
end

function load_o2plus_source_rates(path = data_path("O2plus_source_rates.mat"); fill_invalid = true)
    resolved = resolve_project_path(path)
    data = matread(resolved)
    production_density = Float64.(_mat_value(data, "O2plus_production_density_m3_inv_s_inv"))
    production_rate = Float64.(_mat_value(data, "O2plus_production_rate_s_inv"))
    outflow_rate = Float64.(_mat_value(data, "O2plus_outflow_rate_s_inv"))
    cell_volume = Float64.(_mat_value(data, "cell_volume_m3"))
    surface_area = Float64.(_mat_value(data, "surface_area_m2"))
    r = Float64.(vec(_mat_value(data, "r_m")))
    theta = Float64.(vec(_mat_value(data, "theta_rad")))
    phi = Float64.(vec(_mat_value(data, "phi_rad")))

    fill_invalid && foreach(A -> A[.!isfinite.(A)] .= 0.0,
        (production_density, production_rate, outflow_rate))

    size(production_density) == (length(r), length(theta), length(phi)) ||
        error("O2+ production density grid does not match coordinates")
    size(production_rate) == size(production_density) ||
        error("O2+ production rate grid does not match density grid")
    size(cell_volume) == size(production_density) ||
        error("Source cell volume grid does not match density grid")
    size(outflow_rate) == (length(theta), length(phi)) ||
        error("O2+ outflow grid does not match angular coordinates")
    size(surface_area) == size(outflow_rate) ||
        error("Surface area grid does not match outflow grid")

    metadata = Dict{String, Any}()
    for key in keys(data)
        value = data[key]
        if !(value isa AbstractArray) || length(value) <= 10
            metadata[String(key)] = value
        end
    end

    return O2plusSourceRates(
        resolved, r, theta, phi, production_density, production_rate,
        outflow_rate, cell_volume, surface_area, metadata,
    )
end

function assert_same_grid(source::O2plusSourceRates, fields::MHDFields)
    isapprox(source.r, fields.r; rtol = 1.0e-8, atol = 1.0e-6) ||
        error("Source-rate radial grid does not match MHD grid")
    isapprox(source.theta, fields.theta; rtol = 1.0e-12, atol = 1.0e-12) ||
        error("Source-rate theta grid does not match MHD grid")
    isapprox(source.phi, fields.phi; rtol = 1.0e-12, atol = 1.0e-12) ||
        error("Source-rate phi grid does not match MHD grid")
    return true
end

function _colatitude_cos_widths(theta::AbstractVector{<:Real})
    ntheta = length(theta)
    d_cos_theta = Vector{Float64}(undef, ntheta)
    for j in 1:ntheta
        th_lo = j == 1 ? 0.0 : 0.5 * (Float64(theta[j - 1]) + Float64(theta[j]))
        th_hi = j == ntheta ? Float64(pi) : 0.5 * (Float64(theta[j]) + Float64(theta[j + 1]))
        d_cos_theta[j] = cos(th_lo) - cos(th_hi)
    end
    return d_cos_theta
end

"""
    calculate_cell_volumes(r, theta, phi)

Compute spherical grid cell volumes `V[i, j, k] = r[i]^2 * dr[i] * Δ(cos θ)_j * dphi[k]`
matching the MarsTP source-rate volume convention (central differences in the interior,
single-sided differences at radial and azimuthal endpoints, and exact colatitude patch
widths).
"""
function calculate_cell_volumes(
        r::AbstractVector{<:Real},
        theta::AbstractVector{<:Real},
        phi::AbstractVector{<:Real},
    )
    nr, ntheta, nphi = length(r), length(theta), length(phi)
    dr = Vector{Float64}(undef, nr)
    dr[1] = Float64(r[2] - r[1])
    for i in 2:(nr - 1)
        dr[i] = 0.5 * Float64(r[i + 1] - r[i - 1])
    end
    dr[nr] = Float64(r[nr] - r[nr - 1])

    dphi = Vector{Float64}(undef, nphi)
    dphi[1] = Float64(phi[2] - phi[1])
    for k in 2:(nphi - 1)
        dphi[k] = 0.5 * Float64(phi[k + 1] - phi[k - 1])
    end
    dphi[nphi] = Float64(phi[nphi] - phi[nphi - 1])

    d_cos_theta = _colatitude_cos_widths(theta)
    volumes = Array{Float64, 3}(undef, nr, ntheta, nphi)
    for k in 1:nphi, j in 1:ntheta, i in 1:nr
        volumes[i, j, k] = Float64(r[i])^2 * dr[i] * d_cos_theta[j] * dphi[k]
    end
    return volumes
end

"""
    calculate_surface_areas(r_shell, theta, phi)

Compute spherical boundary patch areas `dA[j, k] = r_shell^2 * Δ(cos θ)_j * dphi_seam[k]`
with half-width azimuthal seam endpoints (`k = 1` and `k = nphi`).
"""
function calculate_surface_areas(
        r_shell::Real,
        theta::AbstractVector{<:Real},
        phi::AbstractVector{<:Real},
    )
    ntheta, nphi = length(theta), length(phi)
    dphi = Vector{Float64}(undef, nphi)
    dphi[1] = 0.5 * Float64(phi[2] - phi[1])
    for k in 2:(nphi - 1)
        dphi[k] = 0.5 * Float64(phi[k + 1] - phi[k - 1])
    end
    dphi[nphi] = 0.5 * Float64(phi[nphi] - phi[nphi - 1])

    d_cos_theta = _colatitude_cos_widths(theta)
    areas = Matrix{Float64}(undef, ntheta, nphi)
    r2 = Float64(r_shell)^2
    for k in 1:nphi, j in 1:ntheta
        areas[j, k] = r2 * d_cos_theta[j] * dphi[k]
    end
    return areas
end

"""
    compute_o2plus_source_rates(mhd_path = data_path("mars_fields_spherical_from_dat.vts");
                                gitm_path = data_path("gitm_sph.mat"),
                                amps_path = data_path("amps_sph.mat"),
                                output_path = nothing)

Compute volumetric O2+ charge-exchange production rates and inner-boundary ionospheric
outflow rates on the MHD spherical grid from `mhd_path`, `gitm_path`, and `amps_path`.
Optionally writes the resulting dictionary to `output_path` in `.mat` format.
"""
function compute_o2plus_source_rates(
        mhd_path = data_path("mars_fields_spherical_from_dat.vts");
        gitm_path = data_path("gitm_sph.mat"),
        amps_path = data_path("amps_sph.mat"),
        output_path = nothing,
    )
    resolved_mhd = resolve_project_path(mhd_path)
    resolved_gitm = resolve_project_path(gitm_path)
    resolved_amps = resolve_project_path(amps_path)

    moments = load_mhd_moments(resolved_mhd)
    gitm = load_gitm(resolved_gitm)
    amps = load_amps(resolved_amps)

    r, theta, phi = moments.r, moments.theta, moments.phi
    ntheta, nphi = length(theta), length(phi)

    prod = o2plus_production_density(moments, gitm, amps, r, theta, phi)
    production_density = Float64.(prod.CO2p_O2p .+ prod.Op_O2p)
    production_density[.!isfinite.(production_density) .| (production_density .< 0.0)] .= 0.0

    cell_volume = calculate_cell_volumes(r, theta, phi)
    production_rate = production_density .* cell_volume

    surface_area = calculate_surface_areas(r[1], theta, phi)
    n_bnd = Float64.(moments.O2plus.n[1, :, :])
    U_bnd = moments.O2plus.v
    vr_bnd = Matrix{Float64}(undef, ntheta, nphi)
    outflow_flux = Matrix{Float64}(undef, ntheta, nphi)
    outflow_rate = Matrix{Float64}(undef, ntheta, nphi)

    for k in 1:nphi, j in 1:ntheta
        sin_th, cos_th = sincos(theta[j])
        sin_ph, cos_ph = sincos(phi[k])
        ux, uy, uz = U_bnd[1, 1, j, k], U_bnd[2, 1, j, k], U_bnd[3, 1, j, k]
        vr = ux * sin_th * cos_ph + uy * sin_th * sin_ph + uz * cos_th
        vr_bnd[j, k] = isfinite(vr) ? vr : 0.0
        nb = isfinite(n_bnd[j, k]) ? max(n_bnd[j, k], 0.0) : 0.0
        n_bnd[j, k] = nb
        flux = nb * max(vr_bnd[j, k], 0.0)
        outflow_flux[j, k] = flux
        outflow_rate[j, k] = flux * surface_area[j, k]
    end

    total_production = sum(production_rate)
    total_outflow = sum(outflow_rate)

    mat_dict = Dict{String, Any}(
        "O2plus_boundary_density_m3_inv" => n_bnd,
        "O2plus_boundary_radial_velocity_m_s" => vr_bnd,
        "O2plus_outflow_flux_m2_inv_s_inv" => outflow_flux,
        "O2plus_outflow_rate_s_inv" => outflow_rate,
        "O2plus_production_density_m3_inv_s_inv" => production_density,
        "O2plus_production_rate_s_inv" => production_rate,
        "altitude_m" => r .- Rm,
        "boundary_altitude_m" => r[1] - Rm,
        "cell_volume_m3" => cell_volume,
        "coordinate_convention" => "MSO; theta colatitude [0,pi]; phi azimuth [0,2pi]; tracing grid Rm=3390 km",
        "mars_radius_m" => Rm,
        "missing_value_convention" => "All source rates finite after in-memory polar moment reconstruction; no repair mask exported",
        "outflow_convention" => "n*max(v dot er,0)*dA; Cartesian MHD velocity; exact angular patch areas; half-width phi seam endpoints",
        "outflow_dimension_order" => "theta, phi",
        "phi_rad" => phi,
        "production_dimension_order" => "r, theta, phi",
        "r_m" => r,
        "rate_units" => "s^-1 (ions per second per source element)",
        "source_files" => [basename(resolved_mhd), basename(resolved_gitm), basename(resolved_amps)],
        "surface_area_m2" => surface_area,
        "theta_rad" => theta,
        "total_outflow_s_inv" => total_outflow,
        "total_production_s_inv" => total_production,
        "volume_convention" => "Existing MarsTrace.calculate_volumes, unchanged; full radial endpoint widths and full phi endpoint widths",
    )

    out_resolved = output_path === nothing ? resolved_mhd : abspath(String(output_path))
    if output_path !== nothing
        mkpath(dirname(out_resolved))
        matwrite(out_resolved, mat_dict)
    end

    metadata = Dict{String, Any}()
    for (k, v) in mat_dict
        if !(v isa AbstractArray) || length(v) <= 10
            metadata[k] = v
        end
    end

    return O2plusSourceRates(
        out_resolved, r, theta, phi, production_density, production_rate,
        outflow_rate, cell_volume, surface_area, metadata,
    )
end
