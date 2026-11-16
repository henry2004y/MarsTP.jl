@testset "1D Analytical Outflow Column & TP-MHD Discrepancy Benchmark" begin
    col = AnalyticalOutflowColumn()
    @test col.species == "O2+"
    @test column_vth(col) ≈ sqrt(MarsTP.TP.kB * 1500.0 / MarsTP.TP.SpeciesDict["O2+"].m)

    # Check source rate and continuity equation dGamma/dz = S_net(z)
    @test column_source_rate(col, 170.0e3) == 0.0
    @test column_source_rate(col, 250.0e3) > 0.0
    @test column_mhd_flux(col, 175.0e3) ≈ col.Gamma_base
    # Ionospheric O2+ production confinement: < 1% of (>200 km) production occurs above 400 km
    prod_above_200 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 200.0e3)
    prod_above_400 = column_mhd_flux(col, col.z_max) - column_mhd_flux(col, 400.0e3)
    @test prod_above_400 / prod_above_200 < 0.01

    z_test = 320.0e3
    eps_z = 10.0
    dgamma_dz = (column_mhd_flux(col, z_test + eps_z) - column_mhd_flux(col, z_test - eps_z)) / (2.0 * eps_z)
    @test dgamma_dz ≈ column_source_rate(col, z_test) rtol = 1e-5
    @test column_mhd_density(col, z_test) * column_mhd_velocity(col, z_test) ≈ column_mhd_flux(col, z_test)

    # Check strict density monotonicity, positive velocity gradient, and exact analytical derivatives
    z_mono_grid = range(col.z_min, col.z_max; length = 151)
    @test all(MarsTP.column_dn_dz(col, z) < 0.0 for z in z_mono_grid)
    @test all(column_mhd_du_dz(col, z) > 0.0 for z in z_mono_grid)
    du_num = (column_mhd_velocity(col, z_test + eps_z) - column_mhd_velocity(col, z_test - eps_z)) / (2.0 * eps_z)
    @test column_mhd_du_dz(col, z_test) ≈ du_num rtol = 1e-6
    dln_n_num = (log(column_mhd_density(col, z_test + eps_z)) - log(column_mhd_density(col, z_test - eps_z))) / (2.0 * eps_z)
    @test column_dln_n_dz(col, z_test) ≈ dln_n_num rtol = 1e-6

    # Check oblique, horizontal, and parallel magnetic and electric field decompositions
    B_ob = column_magnetic_field(col, 400.0e3; b_geometry = :oblique)
    @test B_ob[1] > col.Bx_top
    @test B_ob[2] == 0.0
    @test B_ob[3] == col.Bz0

    B_par = column_magnetic_field(col, 400.0e3; b_geometry = :parallel)
    @test B_par[1] == 0.0 && B_par[2] == 0.0 && B_par[3] > 0.0

    ef_full = column_electric_fields(col, 400.0e3; b_geometry = :oblique, e_mode = :full, include_gradpi = false)
    @test ef_full.total ≈ ef_full.conv + ef_full.hall + ef_full.ambipolar
    # Hall field is strictly perpendicular to B
    @test abs(dot(ef_full.hall, ef_full.B)) < 1e-18
    # Convective field is strictly perpendicular to B
    @test abs(dot(ef_full.conv, ef_full.B)) < 1e-18

    # Vertical MHD momentum balance:
    # q * (E_C + E_H + E_A)_z == m * U_z * dU_z/dz + (1/n) dP_i/dz (i.e. missing -dP_i/dz)
    uz_400 = column_mhd_velocity(col, 400.0e3)
    du_400 = column_mhd_du_dz(col, 400.0e3)
    @test col.q_ion * (ef_full.total[3] - ef_full.ambipolar[3]) ≈
          col.m_ion * uz_400 * du_400 - col.q_ion * ef_full.gradpi[3] rtol = 1e-6

    # Mechanism 1: Boundary volume-Maxwellian vs flux-Maxwellian weighting
    w_res = benchmark_weighting_vdf(col; N_p = 4_000)
    # Flux-weighted reconstructed VDF matches exact MHD Maxwellian down to the lowest bin
    @test isapprox(w_res.f_flux[1], w_res.f_mhd[1]; rtol = 0.10)
    core_mask = 0.25 .<= w_res.v_km_s .<= 1.8
    @test isapprox(w_res.f_flux[core_mask], w_res.f_mhd[core_mask]; rtol = 0.06)
    # Volume-weighted reconstructed VDF overpredicts slow particles (v_z < U_0) and underpredicts fast particles (v_z > U_0)
    slow_idx = findfirst(v -> 0.15 <= v <= 0.25, w_res.v_km_s)
    fast_idx = findfirst(v -> v >= 1.2, w_res.v_km_s)
    @test w_res.f_vol[slow_idx] > 1.5 * w_res.f_mhd[slow_idx]
    @test w_res.f_vol[fast_idx] < 0.8 * w_res.f_mhd[fast_idx]

    # Mechanism 2 & 3: Column tracing & z0 sensitivity (200 vs. 400 vs. 600 km at 800 km)
    r200 = trace_outflow_column(col; z0_km = 200.0, N_p = 800, seed = 11)
    r400 = trace_outflow_column(col; z0_km = 400.0, N_p = 800, seed = 12, record_vdf_600 = true)
    r600 = trace_outflow_column(col; z0_km = 600.0, N_p = 800, seed = 13)
    idx_800 = findfirst(>=(800.0), r200.z_km)
    # Single-shell launch at 200 km misses volumetric production and suffers FLR back-gyration loss
    @test r200.flux_m2_s[idx_800] < 0.35 * r400.flux_m2_s[idx_800]
    # 400 km launch captures ~99% of asymptotic MHD flux while preserving low-altitude energization
    @test isapprox(r400.flux_m2_s[idx_800], column_mhd_flux(col, 800.0e3); rtol = 0.06)
    # 600 km launch omits 400-600 km acceleration and significantly underestimates cumulative work
    @test r600.shares_800.total_eV < 0.65 * r400.shares_800.total_eV
    # Electric-field work shares sum to 100% and convective + Hall dominate over ambipolar
    for r in (r200, r400, r600)
        @test r.shares_800.conv_pct + r.shares_800.hall_pct + r.shares_800.amb_pct ≈ 100.0 atol = 1e-6
    end
    @test r400.shares_800.conv_pct > r400.shares_800.hall_pct > r400.shares_800.amb_pct
    @test r200.shares_600.conv_pct + r200.shares_600.hall_pct + r200.shares_600.amb_pct ≈ 100.0 atol = 1e-6
    @test sum(r400.vdf_600) > 0.0

    # Mechanism 2: Exact 1D parallel-field isothermal expansion (-∇P_e vs -∇P_i)
    p2 = benchmark_parallel_gradpi(col; N_p = 2_000)
    @test abs(p2.u_free_km_s[end] - p2.u_free_km_s[1]) < 0.005
    @test all(p2.u_amb_km_s[3:end] .< p2.u_mhd_km_s[3:end])
    @test p2.u_amb_km_s[end] > p2.u_free_km_s[end] + 0.8
    @test p2.u_amb_km_s[end] < p2.u_mhd_km_s[end] - 0.3
    @test isapprox(p2.u_full_km_s[end], p2.u_mhd_km_s[end]; rtol = 0.005)
end

