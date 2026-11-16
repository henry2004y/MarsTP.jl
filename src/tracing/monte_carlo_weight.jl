# Adapted from ChiZhang123456/MarsASPEN.jl src/monte_carlo_weight.jl,
# Git blob 41d84d67602c5fa284f172a540cccc46531ca8e0.
# MarsTP defaults to O2+ rather than hydrogen. SI except temperature in eV.
Base.@kwdef struct MonteCarloWeight
    sampling_temperature_factor::Float64 = 1.0
    source_number_density_m3::Float64 = 0.0
    unit_particle_weight::Float64 = 1.0
end

_mc_positive(x) = isfinite(x) && x > 0
_mc_nonnegative(x) = isfinite(x) && x >= 0
const _MC_QE = 1.602176634e-19

# log P(Z > a) for standard normal Z, accurate even for large positive a.
function _mc_log_normal_tail(a::Float64)
    if a < 10.0
        return log(0.5 * ccall((:erfc, Base.Math.libm), Cdouble, (Cdouble,), a / sqrt(2.0)))
    end
    z = inv(a * a)
    return -0.5 * a * a - log(a) - 0.5 * log(2.0 * pi) +
           log(1.0 - z + 3.0 * z^2 - 15.0 * z^3 + 105.0 * z^4 - 945.0 * z^5)
end

# Draw Z ~ N(0, 1) conditioned on Z > a using Robert exponential rejection when a > 0.
function _mc_positive_normal(rng::AbstractRNG, a::Float64)
    if a <= 0.0
        while true
            z = randn(rng)
            z > a && return z
        end
    end
    alpha = 0.5 * (a + sqrt(a * a + 4.0))
    while true
        z = a + randexp(rng) / alpha
        rand(rng) <= exp(-0.5 * (z - alpha)^2) && return z
    end
end

"""
    maxwellian_surface_flux(n, u_normal, sigma) -> Float64

Exact analytical one-way outward kinetic flux across a surface normal [m^-2 s^-1]:
`∫_{v_n > 0} v_n n g(v; u_normal, sigma) dv_n`, where `sigma = sqrt(k_B T / m)` [m/s].
"""
function maxwellian_surface_flux(n::Real, u_normal::Real, sigma::Real)
    _mc_nonnegative(n) && isfinite(u_normal) && _mc_nonnegative(sigma) ||
        throw(ArgumentError("finite nonnegative density/sigma and finite u_normal required"))
    nf = Float64(n)
    uf = Float64(u_normal)
    sf = Float64(sigma)
    nf == 0.0 && return 0.0
    sf == 0.0 && return nf * max(uf, 0.0)
    s = uf / sf
    p_pos = exp(_mc_log_normal_tail(-s))
    return max(nf * (uf * p_pos + (sf / sqrt(2.0 * pi)) * exp(-0.5 * s * s)), 0.0)
end

"""Maxwellian thermal speed sqrt(2 kT/m) [m/s]; temperature_ev means kT/e."""
function thermal_speed_from_temperature_ev(temperature_ev::Real,
        mass_kg::Real=TP.SpeciesDict["O2+"].m)
    _mc_positive(temperature_ev) || throw(ArgumentError("temperature must be finite and positive"))
    _mc_positive(mass_kg) || throw(ArgumentError("mass must be finite and positive"))
    sqrt(2 * Float64(temperature_ev) * _MC_QE / Float64(mass_kg))
end

"""Dimensionless normalized Maxwellian ratio g(v;U,T)/gs(v;U,Ts)."""
function maxwellian_importance_weight_3d(U, v, temperature_ev::Real,
        sampled_temperature_ev::Real; mass_kg::Real=TP.SpeciesDict["O2+"].m)
    length(U) == length(v) == 3 && all(isfinite,U) && all(isfinite,v) ||
        throw(ArgumentError("velocities must contain three finite components"))
    a = thermal_speed_from_temperature_ev(temperature_ev,mass_kg)
    b = thermal_speed_from_temperature_ev(sampled_temperature_ev,mass_kg)
    d2 = sum((Float64(v[k])-Float64(U[k]))^2 for k in 1:3)
    exp(3log(b/a) + d2*(inv(b^2)-inv(a^2)))
end

"""Self-normalized density contribution [m^-3]; sum across one source equals n."""
function particle_density_weight(w::Real,n::Real,total::Real)
    _mc_nonnegative(w) && _mc_nonnegative(n) && _mc_positive(total) ||
        throw(ArgumentError("finite nonnegative weight/density and positive total required"))
    Float64(n)*(Float64(w)/Float64(total))
end

"""
    sample_maxwellian_source(N; position_m, bulk_velocity_m_s, temperature_ev,
        species="O2+", weights=MonteCarloWeight(), rng=Random.default_rng(),
        normal=nothing, area_m2=nothing, flux_model=:bulk_speed)

Sample N velocities from an untruncated drifting Maxwellian at Ts=factor*T.
Return initial_states ([x,y,z,vx,vy,vz], m and m/s), dimensionless
importance_weights, density_weights_m3 (nothing for unitless mode), and
macro_weights (density weights or unit_particle_weight*importance_weights).
The Cartesian position and velocity must use the field coordinate frame.

If area_m2 is supplied, the default flux_model=:bulk_speed returns
rate_weights_s = n*A*norm(U)*importance_weight/sum(importance_weights) [s^-1].
This prescribed source injection uses the bulk speed, with no velocity-sign
selection. Sum of patch rates is n*A*norm(U); zero bulk speed gives zero rate.
Self-normalized importance weights have finite-sample bias for distribution
estimates. normal is optional and does not affect bulk-speed rates.
The explicit legacy flux_model=:reservoir requires a normal and returns
n*A*max(v dot normal,0)*importance_weight/N instead.
flux_model=:surface_flux draws conditionally outward velocities (v dot normal > 0)
from the broadened Maxwellian at Ts=factor*T with unbiased one-way kinetic flux
weights n*A*(v dot normal)*P_s(v dot normal > 0)*(g/gs)/N (all particles outward).
Each patch needs its own N and local moments.
No propagation, collisions, escape classification, or files are produced here.
"""
function sample_maxwellian_source(N::Integer; position_m, bulk_velocity_m_s,
        temperature_ev, species="O2+", weights=MonteCarloWeight(),
        rng=Random.default_rng(), normal=nothing, area_m2=nothing, flux_model=:bulk_speed)
    N > 0 || throw(ArgumentError("N must be positive"))
    x,U = SVector{3,Float64}(position_m),SVector{3,Float64}(bulk_velocity_m_s)
    all(isfinite,x) && all(isfinite,U) || throw(ArgumentError("nonfinite state"))
    c,n = weights.sampling_temperature_factor, weights.source_number_density_m3
    _mc_positive(c) && _mc_nonnegative(n) && _mc_nonnegative(weights.unit_particle_weight) ||
        throw(ArgumentError("invalid MonteCarloWeight settings"))
    mass = TP.SpeciesDict[species].m
    sigma = thermal_speed_from_temperature_ev(temperature_ev*c,mass)/sqrt(2)
    sigma0 = thermal_speed_from_temperature_ev(temperature_ev,mass)/sqrt(2)
    flux_model in (:bulk_speed,:reservoir,:surface_flux) || throw(ArgumentError("unknown flux_model"))
    normal !== nothing && area_m2 === nothing && throw(ArgumentError("normal requires area_m2"))
    area_m2 === nothing || _mc_positive(area_m2) || throw(ArgumentError("positive finite area required"))
    flux_model==:reservoir && area_m2!==nothing && normal===nothing &&
        throw(ArgumentError("reservoir rate requires normal"))
    flux_model==:surface_flux && (area_m2===nothing || normal===nothing) &&
        throw(ArgumentError("surface_flux mode requires both area_m2 and normal"))
    er = if normal === nothing
        nothing
    else
        e = SVector{3,Float64}(normal)
        all(isfinite,e) && _mc_positive(norm(e)) ||
            throw(ArgumentError("finite nonzero normal required"))
        normalize(e)
    end
    states = Vector{SVector{6,Float64}}(undef,N)
    w = Vector{Float64}(undef,N)
    rates = area_m2 === nothing ? nothing : zeros(N)
    if flux_model == :surface_flux
        t1 = normalize(cross(abs(er[3]) < 0.9 ? SVector(0.0,0.0,1.0) : SVector(1.0,0.0,0.0), er))
        t2 = cross(er, t1)
        ur = dot(U, er)
        a_s = -ur / sigma
        log_ps = _mc_log_normal_tail(a_s)
        u_t1 = dot(U, t1)
        u_t2 = dot(U, t2)
        for i in 1:N
            vr = ur + sigma * _mc_positive_normal(rng, a_s)
            v = vr * er + (u_t1 + sigma * randn(rng)) * t1 + (u_t2 + sigma * randn(rng)) * t2
            states[i] = SVector{6,Float64}(x..., v...)
            d2 = sum(abs2, v - U)
            wi = exp(1.5 * log(c) + 0.5 * d2 * (inv(sigma^2) - inv(sigma0^2)) + log_ps)
            w[i] = wi
            rates[i] = n * area_m2 * vr * wi / N
        end
    else
        for i in 1:N
            v = U + sigma*SVector{3,Float64}(randn(rng,3))
            states[i] = SVector{6,Float64}(x...,v...)
            w[i] = maxwellian_importance_weight_3d(U,v,temperature_ev,temperature_ev*c; mass_kg=mass)
            if rates !== nothing && flux_model==:reservoir
                rates[i] = n*area_m2*max(dot(v,er),0)*w[i]/N
            end
        end
    end
    total = sum(w)
    _mc_positive(total) && all(isfinite,w) || error("Importance weights underflowed/overflowed; adjust sampling temperature")
    density = n>0 ? particle_density_weight.(w,n,total) : nothing
    if rates !== nothing && flux_model==:bulk_speed
        rates .= (n*area_m2*norm(U)).*(w./total)
    end
    macro_values = density === nothing ? weights.unit_particle_weight*w : density
    return (; initial_states=states, importance_weights=w, density_weights_m3=density,
        macro_weights=macro_values, rate_weights_s=rates, effective_sample_size=total^2/sum(abs2,w),
        species, temperature_ev, sampling_temperature_ev=temperature_ev*c)
end
