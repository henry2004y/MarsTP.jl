# ==============================================================================
# 1D Analytical Ionospheric Outflow Column Benchmark (TP vs. MHD Discrepancy)
# ==============================================================================
#
# Demonstrates why collisionless test-particle (TP) tracing initialized with the
# exact steady-state MHD Maxwellian moments (n, U, T) systematically differs from
# global MHD profiles and exhibits launch-altitude (z_0) sensitivity, while
# preserving relative electric-field work shares (W_C, W_H, W_A).
#
# Uses an oblique draped magnetic field B(z) = (B_x(z), 0, B_z) with sheared
# magnetic pileup B_x(z) so that J = (1/mu_0) dB_x/dz y_hat produces a self-consistent
# Hall electric field E_H = (J x B) / (q n), alongside E_C = -U x B and
# E_A = -(1 / q n) dP_e/dz.

const _MU0 = 4e-7 * π

"""
    AnalyticalOutflowColumn(; kwargs...)

Self-contained 1D analytical Martian dayside ionospheric outflow column on
`z ∈ [z_min, z_max]` (in meters above the surface, without gravity to match
Manuscript Eq. 5).

Contains smooth analytical profiles for:
- Net volumetric ion production `S_net(z) = P(z) - L(z)` [m⁻³ s⁻¹]
- Upward MHD number flux `Γ_MHD(z) = n_MHD(z) * U_z(z)` [m⁻² s⁻¹] satisfying
  `dΓ_MHD/dz = S_net(z)`
- Vertical MHD bulk outflow velocity `U_z(z)` [m/s] and sheared horizontal flow
  `U_y(z)` [m/s] (from the `J × B` magnetic tension force in an oblique draped field)
- Oblique draped magnetic field `B(z) = (B_x(z), 0, B_z)` [T], where `B_x(z)` is
  chosen self-consistently so that magnetic pressure + tension balance the 1D
  MHD vertical momentum equation `m n U_z dU_z/dz = (J × B)_z - d(P_e + P_i)/dz`,
  while the upstream solar-wind motional electric field `E_{C,y} = -U_z(z) B_x(z)`
  plus Hall `E_H = (J × B)/(q n)` and ambipolar `E_A = -(1/q n) dP_e/dz` accelerate
  the ions.
"""
Base.@kwdef struct AnalyticalOutflowColumn
    species::String = "O2+"
    m_ion::Float64 = TP.SpeciesDict["O2+"].m
    q_ion::Float64 = TP.SpeciesDict["O2+"].q
    T_i_K::Float64 = 1500.0
    T_e_K::Float64 = 2000.0
    z_min::Float64 = 150.0e3
    z_max::Float64 = 800.0e3
    z_inner::Float64 = 200.0e3
    Gamma_base::Float64 = 2.0e11
    S0::Float64 = 1.5e7
    z_prod_peak::Float64 = 220.0e3
    H_prod_decay::Float64 = 70.0e3
    z_prod_base::Float64 = 180.0e3
    H_prod_rise::Float64 = 40.0e3
    u_base::Float64 = 300.0
    u_asymp::Float64 = 3700.0
    z_accel::Float64 = 380.0e3
    H_accel::Float64 = 140.0e3
    Bx_top::Float64 = 14.0e-9
    Bz0::Float64 = 5.0e-9
    alpha_hall::Float64 = 0.35
end

"""1D thermal speed `v_th = sqrt(k_B * T_i / m)` [m/s]."""
column_vth(col::AnalyticalOutflowColumn) = sqrt(TP.kB * col.T_i_K / col.m_ion)

"""Net volumetric ion production rate `S_net(z) = P(z) - L(z)` [m⁻³ s⁻¹]."""
function column_source_rate(col::AnalyticalOutflowColumn, z::Real)
    zf = Float64(z)
    zf <= col.z_prod_base && return 0.0
    return col.S0 * exp(-(zf - col.z_prod_peak) / col.H_prod_decay) *
           (1.0 - exp(-(zf - col.z_prod_base) / col.H_prod_rise))
end

"""Exact analytical integral of `column_source_rate` from `z_prod_base` to `z` plus `Gamma_base` [m⁻² s⁻¹]."""
function column_mhd_flux(col::AnalyticalOutflowColumn, z::Real)
    zf = Float64(z)
    zf <= col.z_prod_base && return col.Gamma_base
    H1 = col.H_prod_decay
    H2 = 1.0 / (1.0 / col.H_prod_decay + 1.0 / col.H_prod_rise)
    c0 = exp((col.z_prod_peak - col.z_prod_base) / col.H_prod_decay)
    dz = zf - col.z_prod_base
    integral = col.S0 * c0 * (H1 * (1.0 - exp(-dz / H1)) - H2 * (1.0 - exp(-dz / H2)))
    return col.Gamma_base + integral
end

"""Vertical MHD bulk speed `U_z(z)` [m/s] and its exact vertical derivative `dU_z/dz` [s⁻¹]."""
function column_mhd_velocity(col::AnalyticalOutflowColumn, z::Real)
    zf = Float64(z)
    arg = (zf - col.z_accel) / col.H_accel
    return col.u_base + 0.5 * col.u_asymp * (1.0 + tanh(arg))
end

function column_mhd_du_dz(col::AnalyticalOutflowColumn, z::Real)
    zf = Float64(z)
    arg = (zf - col.z_accel) / col.H_accel
    return 0.5 * (col.u_asymp / col.H_accel) * sech(arg)^2
end

"""MHD ion number density `n_MHD(z) = Γ_MHD(z) / U_z(z)` [m⁻³]."""
function column_mhd_density(col::AnalyticalOutflowColumn, z::Real)
    return column_mhd_flux(col, z) / column_mhd_velocity(col, z)
end

"""Logarithmic density gradient `d(ln n_MHD)/dz` [m⁻¹]."""
function column_dln_n_dz(col::AnalyticalOutflowColumn, z::Real)
    zf = clamp(Float64(z), col.z_min, col.z_max)
    eps_z = 25.0
    n_hi = column_mhd_density(col, zf + eps_z)
    n_lo = column_mhd_density(col, zf - eps_z)
    return (log(n_hi) - log(n_lo)) / (2.0 * eps_z)
end

"""
    column_magnetic_field(col::AnalyticalOutflowColumn, z::Real;
                          b_geometry::Symbol = :oblique) -> SVector{3, Float64}

Evaluate the magnetic field `B(z)` [T] for the 1D outflow column:
- `b_geometry = :oblique` (default): Oblique draped magnetic field `B(z) = (B_x(z), 0, B_z)`
  where `B_x(z)` is chosen self-consistently so that magnetic pressure + tension balance
  the 1D MHD vertical momentum equation `m n U_z dU_z/dz = (J × B)_z - d(P_e + P_i)/dz`.
- `b_geometry = :parallel`: Vertical magnetic field `B(z) = (0, 0, B_z)` along the flow,
  isolating parallel field-aligned acceleration where `E × B = 0`.
- `b_geometry = :horizontal`: Purely horizontal draped field `B(z) = (B_x(z), 0, 0)`.
"""
function column_magnetic_field(col::AnalyticalOutflowColumn, z::Real;
                               b_geometry::Symbol = :oblique)
    b_geometry in (:oblique, :parallel, :horizontal) ||
        throw(ArgumentError("b_geometry must be :oblique, :parallel, or :horizontal"))
    b_geometry == :parallel && return SA[0.0, 0.0, hypot(col.Bx_top, col.Bz0)]
    zf = clamp(Float64(z), col.z_min, col.z_max)
    # Integrate vertical momentum equation downward from z_max where B_x = Bx_top:
    # B_x^2(z) / (2 μ_0) + P_e(z) + P_i(z) + ∫_{z}^{z_max} m n U_z (dU_z/dz') dz' = const
    n_top = column_mhd_density(col, col.z_max)
    n_loc = column_mhd_density(col, zf)
    kT_tot = TP.kB * (col.T_e_K + col.T_i_K)
    N_quad = 16
    dz_q = (col.z_max - zf) / N_quad
    ram_diff = 0.0
    if dz_q > 0
        for k in 1:N_quad
            zq = zf + (k - 0.5) * dz_q
            ram_diff += col.m_ion * column_mhd_flux(col, zq) * column_mhd_du_dz(col, zq) * dz_q
        end
    end
    mag_press_top = col.Bx_top^2 / (2.0 * _MU0)
    delta_P = max(kT_tot * (n_top - n_loc) + ram_diff, 0.0)
    pileup_extra = (8.0e-9)^2 / (2.0 * _MU0) * ((col.z_max - zf) / (col.z_max - col.z_min))^1.5
    Bx = sqrt(2.0 * _MU0 * (mag_press_top + delta_P + pileup_extra))
    Bz = b_geometry == :horizontal ? 0.0 : col.Bz0
    return SA[Bx, 0.0, Bz]
end

"""
    column_electric_fields(col::AnalyticalOutflowColumn, z::Real;
                           e_mode::Symbol=:full,
                           include_gradpi::Bool=false)

Return `(; total, conv, hall, ambipolar, gradpi, B, U_mhd, n_mhd)` at altitude `z` [m]
for the 1D oblique draped column:
- `U_mhd = (0, U_y(z), U_z(z))` where `U_z(z)` is the vertical outflow speed and
  `U_y(z)` is the cross-field plume drift driven by the convective electric field
- `B = (B_x(z), 0, B_z)`
- `E_C = -U_mhd × B = (U_y B_z, -U_z B_x, -U_y B_x)`: includes both the horizontal
  motional field `-U_z(z) B_x(z) y_hat` that drives `E × B` vertical outflow and the
  vertical convective plume field `(1 - alpha_hall) (m/q) U_z dU_z/dz` from horizontal
  cross-field drift `U_y(z) < 0`
- `E_H = (J × B) / (q n)`: Hall electric field providing fraction `alpha_hall` of the
  macroscopic MHD inertial acceleration plus the magnetic pressure balance against
  thermal pressure gradients
- `E_A = -(1 / (q n)) ∇P_e = (0, 0, -(k_B T_e / q) d(ln n)/dz)`: ambipolar field
- Optional effective fluid-closure force `E_gradpi = -(1 / (q n)) ∇P_i` when
  `include_gradpi=true` (representing collisional conversion of ion thermal enthalpy).

### `e_mode` options:
- `:conv_only`: only `E_C = (0, -U_z B_x, 0)` (no parallel/vertical force `E_H + E_A` along `B_z`)
- `:conv_amb`: `E_C + E_A` (includes ambipolar `-∇P_e/(qn)` but omits Hall `J × B / (qn)`)
- `:full` (default): full generalized Ohm's law `E = E_C + E_H + E_A` (missing only `-∇P_i/(qn)` unless `include_gradpi=true`).
"""
function column_electric_fields(col::AnalyticalOutflowColumn, z::Real;
                                b_geometry::Symbol = :oblique,
                                e_mode::Symbol = :full,
                                include_gradpi::Bool = false)
    e_mode in (:conv_only, :conv_amb, :full) ||
        throw(ArgumentError("e_mode must be :conv_only, :conv_amb, or :full"))
    zf = clamp(Float64(z), col.z_min, col.z_max)
    uz = column_mhd_velocity(col, zf)
    du_dz = column_mhd_du_dz(col, zf)
    n_loc = column_mhd_density(col, zf)
    dln_n = column_dln_n_dz(col, zf)
    B = column_magnetic_field(col, zf; b_geometry = b_geometry)
    Bx, Bz = B[1], B[3]

    # Macroscopic inertial acceleration force per unit charge: (m/q) U_z dU_z/dz
    f_inertial = (col.m_ion / col.q_ion) * (uz * du_dz)

    # Ambipolar electric field: E_A = -(1 / q n) dP_e/dz z_hat
    Ez_amb = -(TP.kB * col.T_e_K / col.q_ion) * dln_n
    E_amb = SA[0.0, 0.0, Ez_amb]

    # Effective ion pressure gradient force -(1 / q n) dP_i/dz z_hat (collisional fluid closure)
    Ez_gradpi = -(TP.kB * col.T_i_K / col.q_ion) * dln_n

    if b_geometry == :parallel
        E_conv = SA[0.0, 0.0, 0.0]
        E_hall = SA[0.0, 0.0, 0.0]
        # In a parallel field where J x B = 0, full_fluid closure requires (m/q) U_z dU_z/dz - E_A
        E_gradpi_vec = include_gradpi ? SA[0.0, 0.0, f_inertial - Ez_amb] : SA[0.0, 0.0, 0.0]
        E_tot = (e_mode == :conv_only) ? E_conv :
                (e_mode == :conv_amb)  ? (E_conv + E_amb) :
                (E_conv + E_amb + E_gradpi_vec)
        return (;
            total = E_tot,
            conv = E_conv,
            hall = E_hall,
            ambipolar = (e_mode == :conv_only ? SA[0.0, 0.0, 0.0] : E_amb),
            gradpi = SA[0.0, 0.0, Ez_gradpi],
            B = B,
            U_mhd = SA[0.0, 0.0, uz],
            n_mhd = n_loc,
        )
    end

    # In the oblique/horizontal draped field, the total MHD fluid vertical force balance is:
    # q (E_C + E_H + E_A)_z - (1/n) dP_i/dz = m U_z dU_z/dz
    # where E_C contributes (1 - alpha_hall) * f_inertial (via plume cross-field drift U_y < 0)
    # and E_H contributes alpha_hall * f_inertial + (k_B T_i / q) dln_n/dz (canceling -dP_i/dz in MHD)
    Ez_conv = (1.0 - col.alpha_hall) * f_inertial
    uy = -Ez_conv / Bx
    Ex_conv = uy * Bz
    Ey_conv = -uz * Bx
    E_conv = (e_mode == :conv_only) ? SA[0.0, Ey_conv, 0.0] : SA[Ex_conv, Ey_conv, Ez_conv]

    Ez_hall = col.alpha_hall * f_inertial - Ez_gradpi
    Ex_hall = -Ez_hall * (Bz / Bx)
    E_hall = SA[Ex_hall, 0.0, Ez_hall]

    E_gradpi = include_gradpi ? SA[0.0, 0.0, Ez_gradpi] : SA[0.0, 0.0, 0.0]

    E_tot = if e_mode == :conv_only
        E_conv
    elseif e_mode == :conv_amb
        E_conv + E_amb
    else
        E_conv + E_hall + E_amb + E_gradpi
    end

    return (;
        total = E_tot,
        conv = E_conv,
        hall = (e_mode == :full ? E_hall : SA[0.0, 0.0, 0.0]),
        ambipolar = (e_mode == :conv_only ? SA[0.0, 0.0, 0.0] : E_amb),
        gradpi = SA[0.0, 0.0, Ez_gradpi],
        B = B,
        U_mhd = SA[0.0, uy, uz],
        n_mhd = n_loc,
    )
end

"""
    upward_maxwellian_flux(n0::Real, u0::Real, v_th::Real) -> Float64

Exact one-way upward kinetic flux `∫_0^∞ v_z f_M(v_z; n0, u0, v_th) dv_z` [m⁻² s⁻¹]
for a 1D drifting Maxwellian with number density `n0`, drift `u0`, and thermal
speed `v_th = sqrt(k_B T / m)`:
`Γ_+ = n0 * (u0/2 * (1 + erf(u0 / (sqrt(2) v_th))) + v_th / sqrt(2π) * exp(-u0^2 / (2 v_th^2)))`.
"""
function upward_maxwellian_flux(n0::Real, u0::Real, v_th::Real)
    vmax = max(Float64(u0), 0.0) + 8.0 * Float64(v_th)
    N_int = 600
    dv = vmax / N_int
    acc = 0.0
    for i in 1:N_int
        v = (i - 0.5) * dv
        acc += v * exp(-0.5 * ((v - u0) / v_th)^2) * dv
    end
    return Float64(n0) * acc / (sqrt(2.0 * π) * Float64(v_th))
end

function _sample_vz_flux(rng::AbstractRNG, u0::Float64, v_th::Float64)
    v_peak = 0.5 * (u0 + sqrt(u0^2 + 4.0 * v_th^2))
    f_max = v_peak * exp(-0.5 * ((v_peak - u0) / v_th)^2)
    v_max = max(u0, 0.0) + 6.5 * v_th
    while true
        vz = rand(rng) * v_max
        if rand(rng) * f_max <= vz * exp(-0.5 * ((vz - u0) / v_th)^2)
            return vz
        end
    end
end

function _sample_vz_volume_upward(rng::AbstractRNG, u0::Float64, v_th::Float64)
    while true
        vz = u0 + v_th * randn(rng)
        vz > 0.0 && return vz
    end
end

"""
    benchmark_weighting_vdf(col::AnalyticalOutflowColumn = AnalyticalOutflowColumn();
                            N_p::Integer = 40_000, u0_m_s::Real = 600.0,
                            n0_m3::Real = 1.0e9)

Demonstrate Mechanism 1 (Manuscript Eqs. 2–4 & 6): in a steady uniform flow of
speed `u0_m_s`, weighting boundary-injected particles by the volume Maxwellian
`w_i ∝ g(v)` gives a residence-time reconstructed PSD `f_TP(v_z) ∝ g(v_z)/v_z`,
whereas flux weighting `w_i ∝ v_z g(v_z)` exactly recovers `f_MHD(v_z)`.
Traces particles using `TestParticle.jl` (`Boris()`) and reconstructs the 1D VDF
via detector residence time.
"""
function benchmark_weighting_vdf(
        col::AnalyticalOutflowColumn = AnalyticalOutflowColumn();
        N_p::Integer = 40_000,
        u0_m_s::Real = 600.0,
        n0_m3::Real = 1.0e9,
    )
    n0 = Float64(n0_m3)
    u0 = Float64(u0_m_s)
    v_th = column_vth(col)
    z0 = 0.0
    z_max = 50.0e3
    dz_det = 10.0e3

    vmin, vmax = 25.0, 3000.0
    v_init = collect(range(vmin, vmax; length = N_p))
    dv_grid = step(range(vmin, vmax; length = N_p))

    # Manuscript Eqs. 2-4: volume-Maxwellian weights w_i ∝ g(v), sum(w_i) = n0 * u0
    g_vol = [exp(-0.5 * ((v - u0) / v_th)^2) for v in v_init]
    w_vol = (n0 * u0) .* (g_vol ./ sum(g_vol))

    # Exact kinetic flux weights: w_i ∝ v * g(v), sum(w_i) = ∫_0^∞ v f_M(v) dv
    f_exact_grid = [(n0 / (sqrt(2.0 * π) * v_th)) * g for g in g_vol]
    Q_flux = sum(v_init .* f_exact_grid .* dv_grid)
    g_flux = v_init .* g_vol
    w_flux = Q_flux .* (g_flux ./ sum(g_flux))

    # Trace in uniform parallel magnetic field B = B_0 z_hat, E = 0 so v_z is conserved
    # and the detector residence time is exactly tau_i = dz_det / v_z,i, isolating
    # the boundary weighting bias (Manuscript Eqs. 2-4 vs Eq. 6) without FLR gyromotion.
    B0 = SA[0.0, 0.0, 15.0e-9]
    E0 = SA[0.0, 0.0, 0.0]
    sp = TP.Ion(32, 1)
    param = TP.prepare(; E = _ -> E0, B = _ -> B0, species = sp)
    u0_list = [SA[0.0, 0.0, z0, 0.0, 0.0, v] for v in v_init]
    prob_func = (prob, ctx) -> TP.remake(prob; u0 = u0_list[ctx.sim_id])
    prob = TP.TraceProblem(u0_list[1], (0.0, 1200.0), param; prob_func = prob_func)
    isoutside = (u, p, t) -> (u[3] < z0 || u[3] >= z_max)
    sols = TP.solve(prob, Boris(), EnsembleThreads(); dt = 0.5, trajectories = N_p, isoutside = isoutside)

    N_vbins = 60
    v_edges = range(0.0, 3000.0; length = N_vbins + 1)
    dv_bin = step(v_edges)
    v_centers = [0.5 * (v_edges[k] + v_edges[k + 1]) for k in 1:N_vbins]
    f_rec_vol = zeros(Float64, N_vbins)
    f_rec_flux = zeros(Float64, N_vbins)

    for (i, sol) in enumerate(sols.u)
        u_vec = sol.u
        t_vec = sol.t
        tau_det = 0.0
        for k in 1:(length(u_vec) - 1)
            z_mid = 0.5 * (u_vec[k][3] + u_vec[k + 1][3])
            if 20.0e3 <= z_mid < 30.0e3
                tau_det += t_vec[k + 1] - t_vec[k]
            end
        end
        if tau_det > 0.0
            v_bin = clamp(Int(floor(v_init[i] / dv_bin)) + 1, 1, N_vbins)
            f_rec_vol[v_bin] += w_vol[i] * tau_det / (dz_det * dv_bin)
            f_rec_flux[v_bin] += w_flux[i] * tau_det / (dz_det * dv_bin)
        end
    end

    f_mhd_bins = [(n0 / (sqrt(2.0 * π) * v_th)) * exp(-0.5 * ((v - u0) / v_th)^2) for v in v_centers]
    # Analytical bias curve (n0 * u0 / ∫ g dv) * g(v) / v
    norm_vol = (n0 * u0) / sum(g_vol .* dv_grid)
    f_analytic_bias = [v >= vmin ? norm_vol * exp(-0.5 * ((v - u0) / v_th)^2) / v : 0.0 for v in v_centers]

    return (;
        v_km_s = v_centers ./ 1.0e3,
        f_mhd = f_mhd_bins,
        f_vol = f_rec_vol,
        f_flux = f_rec_flux,
        f_analytic_bias,
    )
end

"""
    trace_outflow_column(col::AnalyticalOutflowColumn = AnalyticalOutflowColumn();
                         z0_km::Real = 400.0,
                         sample_mode::Symbol = :volume_full,
                         volumetric_source::Bool = false,
                         include_gradpi::Bool = false,
                         flr_scale::Real = 1.0,
                         N_p::Integer = 12_000,
                         N_bins::Integer = 130,
                         dt_step::Real = 0.2,
                         seed::Integer = 42,
                         record_vdf_600::Bool = false)

Trace `N_p` `O₂⁺` test particles through the 1D oblique draped column `col`
using `TestParticle.jl` (`Boris()`, `EnsembleThreads()`) and reconstruct:
- Altitude profiles of density `n(z)`, vertical flux `Γ_z(z)`, and bulk speed `U_z(z)`
- Residence-time weighted cumulative electric work profiles `W_C(z)`, `W_H(z)`, `W_A(z)`, `W_tot(z)` [eV]
- Optional 2D `(v_y, v_z)` velocity distribution function at `z ∈ [580, 620]` km.

### Modes for Progressive Closure & `z_0` Sensitivity
- `sample_mode = :volume_full`: Manuscript Eqs. 2–4 untruncated 3D drifting Maxwellian
  at `z_0` with total injection rate `Q = n(z_0) * U_z(z_0)`. Particles with initial
  downward motion or downward first-gyration that cross `z < z_inner` (200 km) are absorbed.
- `sample_mode = :volume_upward`: Upward-only volume-weighted Maxwellian (`v_z > 0`)
  with `Q = n(z_0) * U_z(z_0)`.
- `sample_mode = :flux`: Exact one-way upward kinetic flux weighting `w_i ∝ v_z g(v)`
  with `Q = ∫_{v_z>0} v_z f_M(v) d³v`.
- `volumetric_source = true`: Inject particles at both `z_0` (with flux weighting) and
  continuously across `z ∈ [z_0, z_max]` proportional to `S_net(z) = P(z) - L(z)`.
- `include_gradpi = true`: Include the effective fluid-closure ion pressure gradient
  force `-(1 / q n) dP_i/dz` alongside `E_C + E_H + E_A`.
- `flr_scale`: Scales the effective gyro-radius `r_L -> flr_scale * r_L` by scaling
  `(q/m) -> (q/m) / flr_scale` in the Lorentz equation (so `flr_scale = 1.0` is exact
  full-orbit `O₂⁺`, and `flr_scale = 0.1` approaches the small-gyroradius MHD fluid limit).
"""
function trace_outflow_column(
        col::AnalyticalOutflowColumn = AnalyticalOutflowColumn();
        z0_km::Real = 400.0,
        b_geometry::Symbol = :oblique,
        e_mode::Symbol = :full,
        sample_mode::Symbol = :volume_full,
        volumetric_source::Bool = false,
        include_gradpi::Bool = false,
        flr_scale::Real = 1.0,
        N_p::Integer = 12_000,
        N_bins::Integer = 130,
        dt_step::Real = 0.2,
        seed::Integer = 42,
        record_vdf_600::Bool = false,
    )
    sample_mode in (:volume_full, :volume_upward, :flux) ||
        throw(ArgumentError("sample_mode must be :volume_full, :volume_upward, or :flux"))
    b_geometry in (:oblique, :parallel, :horizontal) ||
        throw(ArgumentError("b_geometry must be :oblique, :parallel, or :horizontal"))
    e_mode in (:conv_only, :conv_amb, :full) ||
        throw(ArgumentError("e_mode must be :conv_only, :conv_amb, or :full"))
    flr_scale > 0 || throw(ArgumentError("flr_scale must be positive"))

    rng = Random.Xoshiro(seed)
    z_min, z_max, z_inner = col.z_min, col.z_max, col.z_inner
    z_edges = range(z_min, z_max; length = N_bins + 1)
    dz_bin = step(z_edges)
    z_centers = [0.5 * (z_edges[k] + z_edges[k + 1]) for k in 1:N_bins]

    z0 = Float64(z0_km) * 1.0e3
    n0 = column_mhd_density(col, z0)
    u0 = column_mhd_velocity(col, z0)
    v_th = column_vth(col)

    flux_base = if volumetric_source
        # In the exact MHD continuity equation dGamma/dz = S_net(z), the base boundary flux
        # entering at z0 is Gamma_MHD(z0) = n0 * u0
        n0 * u0
    elseif sample_mode == :flux
        upward_maxwellian_flux(n0, u0, v_th)
    else
        # Manuscript Eq. 3: Q = n * |U| * A (per unit area A = 1 m^2)
        n0 * u0
    end

    vol_flux = volumetric_source ? (column_mhd_flux(col, z_max) - column_mhd_flux(col, z0)) : 0.0
    total_injected_flux = flux_base + vol_flux
    w_p = total_injected_flux / N_p

    z_src_grid = range(z0, z_max; length = 250)
    cdf_src = [
        (flux_base + (column_mhd_flux(col, zs) - column_mhd_flux(col, z0))) / total_injected_flux
        for zs in z_src_grid
    ]

    # Pre-tabulate B(z) and E(z) on a fine 1D grid for fast, allocation-free evaluation
    N_tab = 651
    z_tab = range(z_min, z_max; length = N_tab)
    dz_tab = step(z_tab)
    B_tab = Vector{SVector{3, Float64}}(undef, N_tab)
    E_tot_tab = Vector{SVector{3, Float64}}(undef, N_tab)
    E_conv_tab = Vector{SVector{3, Float64}}(undef, N_tab)
    E_hall_tab = Vector{SVector{3, Float64}}(undef, N_tab)
    E_amb_tab = Vector{SVector{3, Float64}}(undef, N_tab)

    for k in 1:N_tab
        ef = column_electric_fields(col, z_tab[k]; b_geometry = b_geometry, e_mode = e_mode, include_gradpi = include_gradpi)
        B_tab[k] = ef.B
        E_tot_tab[k] = ef.total
        E_conv_tab[k] = ef.conv
        E_hall_tab[k] = ef.hall
        E_amb_tab[k] = ef.ambipolar
    end

    @inline function interp_vec(tab::Vector{SVector{3, Float64}}, z_val::Real)
        zc = clamp(Float64(z_val), z_min, z_max)
        pos = (zc - z_min) / dz_tab + 1.0
        idx = clamp(Int(floor(pos)), 1, N_tab - 1)
        frac = pos - idx
        @inbounds return (1.0 - frac) * tab[idx] + frac * tab[idx + 1]
    end

    B_func = r -> interp_vec(B_tab, r[3])
    E_func = r -> interp_vec(E_tot_tab, r[3])

    # Scale effective q/m by 1/flr_scale so r_L -> flr_scale * r_L while keeping E and B physical
    sp_eff = TP.SpeciesDict[col.species]
    param = TP.prepare(; E = E_func, B = B_func, q = col.q_ion / Float64(flr_scale), m = col.m_ion, species = sp_eff)

    u0_list = Vector{SVector{6, Float64}}(undef, N_p)
    for i in 1:N_p
        if volumetric_source
            r_u = rand(rng)
            if r_u <= cdf_src[1]
                z_init = z0
                u_init = u0
                vz = (sample_mode == :flux) ? _sample_vz_flux(rng, u_init, v_th) :
                     (sample_mode == :volume_upward) ? _sample_vz_volume_upward(rng, u_init, v_th) :
                     (u_init + v_th * randn(rng))
            else
                idx_s = clamp(searchsortedfirst(cdf_src, r_u), 2, length(z_src_grid))
                z_init = z_src_grid[idx_s]
                u_init = column_mhd_velocity(col, z_init)
                # Newly photoionized / created ions in fluid rest frame centered at local U(z)
                vz = (flr_scale < 0.5) ? u_init : (u_init + v_th * randn(rng))
            end
        else
            z_init = z0
            vz = if flr_scale < 0.5
                u0
            elseif sample_mode == :flux
                _sample_vz_flux(rng, u0, v_th)
            elseif sample_mode == :volume_upward
                _sample_vz_volume_upward(rng, u0, v_th)
            else
                u0 + v_th * randn(rng)
            end
        end
        v_th_perp = (flr_scale < 0.5) ? 0.0 : v_th
        vx = v_th_perp * randn(rng)
        vy = v_th_perp * randn(rng)
        u0_list[i] = SA[0.0, 0.0, z_init, vx, vy, vz]
    end

    dt_actual = Float64(dt_step) * min(Float64(flr_scale), 1.0)
    tspan = (0.0, 1200.0)
    prob_func = (prob, ctx) -> TP.remake(prob; u0 = u0_list[ctx.sim_id])
    prob = TP.TraceProblem(u0_list[1], tspan, param; prob_func = prob_func)
    isoutside = (u, p, t) -> (u[3] < z_inner || u[3] >= z_max)

    sols = TP.solve(
        prob, Boris(), EnsembleThreads();
        dt = dt_actual, trajectories = N_p, isoutside = isoutside,
    )

    density_accum = zeros(Float64, N_bins)
    flux_z_accum = zeros(Float64, N_bins)
    tau_weight_accum = zeros(Float64, N_bins)
    w_conv_accum = zeros(Float64, N_bins)
    w_hall_accum = zeros(Float64, N_bins)
    w_amb_accum = zeros(Float64, N_bins)
    w_tot_accum = zeros(Float64, N_bins)

    Nv2 = 55
    vy_edges = range(-3500.0, 2500.0; length = Nv2 + 1)
    vz_edges = range(-500.0, 6500.0; length = Nv2 + 1)
    dvy = step(vy_edges)
    dvz = step(vz_edges)
    vdf_600 = zeros(Float64, Nv2, Nv2)

    q_eV = col.q_ion / TP.eV
    n_escaped = 0

    for sol in sols.u
        u_vec = sol.u
        t_vec = sol.t
        n_steps = length(u_vec)
        cum_wc = 0.0
        cum_wh = 0.0
        cum_wa = 0.0
        cum_wt = 0.0

        for k in 1:(n_steps - 1)
            u_a = u_vec[k]
            u_b = u_vec[k + 1]
            z_old = u_a[3]
            z_new = u_b[3]
            dt_s = t_vec[k + 1] - t_vec[k]

            if z_new < z_inner || dt_s <= 0.0
                break
            end
            if z_new >= z_max
                n_escaped += 1
                break
            end

            z_mid = 0.5 * (z_old + z_new)
            v_mid = SA[
                0.5 * (u_a[4] + u_b[4]),
                0.5 * (u_a[5] + u_b[5]),
                0.5 * (u_a[6] + u_b[6]),
            ]

            Ec = interp_vec(E_conv_tab, z_mid)
            Eh = interp_vec(E_hall_tab, z_mid)
            Ea = interp_vec(E_amb_tab, z_mid)
            Et = interp_vec(E_tot_tab, z_mid)

            dwc = q_eV * dot(Ec, v_mid) * dt_s
            dwh = q_eV * dot(Eh, v_mid) * dt_s
            dwa = q_eV * dot(Ea, v_mid) * dt_s
            dwt = q_eV * dot(Et, v_mid) * dt_s

            wc_mid = cum_wc + 0.5 * dwc
            wh_mid = cum_wh + 0.5 * dwh
            wa_mid = cum_wa + 0.5 * dwa
            wt_mid = cum_wt + 0.5 * dwt

            cum_wc += dwc
            cum_wh += dwh
            cum_wa += dwa
            cum_wt += dwt

            bin_idx = clamp(Int(floor((z_mid - z_min) / dz_bin)) + 1, 1, N_bins)
            wt_tau = w_p * dt_s
            density_accum[bin_idx] += wt_tau / dz_bin
            flux_z_accum[bin_idx] += wt_tau * v_mid[3] / dz_bin
            tau_weight_accum[bin_idx] += wt_tau
            w_conv_accum[bin_idx] += wt_tau * wc_mid
            w_hall_accum[bin_idx] += wt_tau * wh_mid
            w_amb_accum[bin_idx] += wt_tau * wa_mid
            w_tot_accum[bin_idx] += wt_tau * wt_mid

            if record_vdf_600 && (580.0e3 <= z_mid <= 620.0e3)
                iy = Int(floor((v_mid[2] - first(vy_edges)) / dvy)) + 1
                iz = Int(floor((v_mid[3] - first(vz_edges)) / dvz)) + 1
                if 1 <= iy <= Nv2 && 1 <= iz <= Nv2
                    vdf_600[iy, iz] += wt_tau / (40.0e3 * dvy * dvz)
                end
            end
        end
    end

    u_rec = [density_accum[k] > 1.0e5 ? flux_z_accum[k] / density_accum[k] : NaN for k in 1:N_bins]
    w_conv_eV = [tau_weight_accum[k] > 0 ? w_conv_accum[k] / tau_weight_accum[k] : NaN for k in 1:N_bins]
    w_hall_eV = [tau_weight_accum[k] > 0 ? w_hall_accum[k] / tau_weight_accum[k] : NaN for k in 1:N_bins]
    w_amb_eV  = [tau_weight_accum[k] > 0 ? w_amb_accum[k] / tau_weight_accum[k] : NaN for k in 1:N_bins]
    w_tot_eV  = [tau_weight_accum[k] > 0 ? w_tot_accum[k] / tau_weight_accum[k] : NaN for k in 1:N_bins]

    idx_600 = findfirst(z -> z >= 600.0e3, z_centers)
    tot_600 = w_tot_eV[idx_600]
    shares_600 = if isfinite(tot_600) && abs(tot_600) > 1e-6
            idx_start = findfirst(>=(z0), z_centers)
            wc_rel = idx_start !== nothing && isfinite(w_conv_eV[idx_start]) ? w_conv_eV .- w_conv_eV[idx_start] : w_conv_eV
            wh_rel = idx_start !== nothing && isfinite(w_hall_eV[idx_start]) ? w_hall_eV .- w_hall_eV[idx_start] : w_hall_eV
            wa_rel = idx_start !== nothing && isfinite(w_amb_eV[idx_start])  ? w_amb_eV  .- w_amb_eV[idx_start]  : w_amb_eV
            wt_rel = idx_start !== nothing && isfinite(w_tot_eV[idx_start])  ? w_tot_eV  .- w_tot_eV[idx_start]  : w_tot_eV

            idx_400 = findfirst(>=(400.0e3), z_centers)
            idx_750 = findfirst(>=(750.0e3), z_centers)
            dw_tot_400_750 = w_tot_eV[idx_750] - w_tot_eV[idx_400]
            shares_400_750 = if isfinite(dw_tot_400_750) && abs(dw_tot_400_750) > 1e-6
                (;
                    conv_pct = 100.0 * (w_conv_eV[idx_750] - w_conv_eV[idx_400]) / dw_tot_400_750,
                    hall_pct = 100.0 * (w_hall_eV[idx_750] - w_hall_eV[idx_400]) / dw_tot_400_750,
                    amb_pct  = 100.0 * (w_amb_eV[idx_750]  - w_amb_eV[idx_400])  / dw_tot_400_750,
                    delta_total_eV = dw_tot_400_750,
                )
            else
                (; conv_pct = NaN, hall_pct = NaN, amb_pct = NaN, delta_total_eV = NaN)
            end

            (;
                conv_pct = 100.0 * w_conv_eV[idx_600] / tot_600,
                hall_pct = 100.0 * w_hall_eV[idx_600] / tot_600,
                amb_pct  = 100.0 * w_amb_eV[idx_600] / tot_600,
                total_eV = tot_600,
                upper_conv_pct = shares_400_750.conv_pct,
                upper_hall_pct = shares_400_750.hall_pct,
                upper_amb_pct  = shares_400_750.amb_pct,
                upper_delta_eV = shares_400_750.delta_total_eV,
            )
    else
        (;
            conv_pct = NaN, hall_pct = NaN, amb_pct = NaN, total_eV = NaN,
            upper_conv_pct = NaN, upper_hall_pct = NaN, upper_amb_pct = NaN, upper_delta_eV = NaN,
        )
    end

    return (;
        z_km = z_centers ./ 1.0e3,
        n_m3 = density_accum,
        flux_m2_s = flux_z_accum,
        u_z_m_s = u_rec,
        w_conv_eV,
        w_hall_eV,
        w_amb_eV,
        w_tot_eV,
        shares_600,
        escape_fraction = n_escaped / N_p,
        total_injected_flux,
        vdf_600,
        vy_centers_km_s = [0.5 * (vy_edges[k] + vy_edges[k + 1]) / 1.0e3 for k in 1:Nv2],
        vz_centers_km_s = [0.5 * (vz_edges[k] + vz_edges[k + 1]) / 1.0e3 for k in 1:Nv2],
    )
end

