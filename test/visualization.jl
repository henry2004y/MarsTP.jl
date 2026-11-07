using JLD2, JSON, PyPlot, StaticArrays, Test

@testset "Mars BS/MPB conics and PyPlot trajectory visualization" begin
    for name in keys(MODELS)
        for ((x, rho), (x0, length_par, ecc)) in zip(boundary_profile(name), MODELS[name])
            @test isapprox(sqrt.((x .- x0).^2 .+ rho.^2) .+ ecc .* (x .- x0), fill(length_par, length(x)); atol = 1e-12)
            @test all(>=(0.0), rho)
        end
    end

    fig, ax = PyPlot.plt.subplots()
    circles = bs_mpb(ax; plane = "YZ", x_slice = 0.0)
    @test length(circles) == 2
    PyPlot.plt.close(fig)

    folder = mktempdir(; cleanup = false)
    path = joinpath(folder, "trajectories_1.jld2")
    jldopen(path, "w") do f
        f["format_version"] = 1
        f["complete"] = true
        f["species"] = "O2+"
        f["coordinate_system"] = "MSO"
        state = zeros(Float64, 7, 4)
        state[1, :] .= 0.0:3.0
        state[2, :] .= (0.0:3.0) .* Rm
        f["p1/state"] = state
        f["p2/state"] = state
        f["p2/species"] = "O+"
    end

    records = load_trajectories(folder; species = "O2+", max_points = 2)
    @test length(records) == 1
    @test records[1]["points"][:, 1] ≈ [0.0, 3.0]
    @test load_trajectories(folder; species = "O+")[1]["id"] == 2
    @test_throws ArgumentError load_trajectories(folder; species = "H+")

    jsonl_path = joinpath(folder, "backward.jsonl")
    open(jsonl_path, "w") do io
        println(io, JSON.json(Dict("id" => 1, "points" => [[0, 0, 2], [1, 0, 2]], "times_s" => [0, -1])))
    end
    @test_throws ArgumentError load_trajectories(jsonl_path; position_unit = "Rm", species = "O2+")
    back = load_trajectories(jsonl_path; position_unit = "Rm", species = "O2+", assume_species = "O2+")
    @test back[1]["times_s"][end] == -1.0

    smoke_png = joinpath(folder, "smoke.png")
    fig, axes, _ = plot_trajectory(
        path;
        planes = ("XZ", "XY", "YZ", "3D"),
        x_slice = 0.0,
        output = smoke_png,
    )
    @test length(axes) == 4
    @test isfile(smoke_png)
    PyPlot.plt.close(fig)

    incomplete_path = joinpath(folder, "trajectories_incomplete.jld2")
    jldopen(incomplete_path, "w") do f
        f["format_version"] = 1
        f["complete"] = false
        f["species"] = "O2+"
    end
    @test_throws ArgumentError load_trajectories(incomplete_path)
end

@testset "TestParticle v0.26 work channels and VDF helpers" begin
    sp = MarsTP.TP.SpeciesDict["O2+"]
    E = SA[0.0, 0.0, 1.0e-5]
    B = SA[0.0, 0.0, 1.0e-8]
    param = MarsTP.TP.prepare(; E = _ -> E, B = _ -> B, species = sp)
    a = sp.q * E[3] / sp.m
    t = [0.0, 0.5, 1.0]
    u = [SA[0.0, 0.0, 0.5 * a * s^2, 0.0, 0.0, a * s] for s in t]
    sol = (; t, u)

    ad = adiabatic_work_profile(sol, param)
    @test ad.summary.parallel_eV ≈ 0.5 * sp.m * (a * last(t))^2 / MarsTP.TP.eV rtol = 1e-10
    @test ad.summary.fermi_eV ≈ 0.0 atol = 1e-12
    @test ad.summary.betatron_eV ≈ 0.0 atol = 1e-12

    fields5 = MarsTP.FieldWorkInterpolators(
        _ -> E,
        _ -> 0.5 * E,
        _ -> 0.25 * E,
        _ -> 0.15 * E,
        _ -> 0.10 * E,
    )
    w5 = field_work(sol, fields5)
    @test w5.total_eV ≈ w5.conv_eV + w5.hall_eV + w5.gradpe_eV + w5.res_eV
    p5 = Electric_field_work_profile(sol, fields5)
    @test maximum(abs, p5.field_sum_residual_eV) < 1e-12

    cfg = BacktraceConfig(
        vx_min_kms = -100.0, vx_max_kms = 100.0,
        vy_min_kms = -100.0, vy_max_kms = 100.0,
        vz_min_kms = -100.0, vz_max_kms = 100.0,
        dv_kms = 10.0, dvy_kms = 10.0,
    )
    axes_v = velocity_axes(cfg)
    vx, vy, vz = axes_v.vx, axes_v.vy, axes_v.vz
    vth = 15.0e3
    u0 = SA[10.0e3, -5.0e3, 20.0e3]
    n0 = 1.0e6
    f = [
        n0 / (pi^1.5 * vth^3) * exp(-((vx[i] - u0[1])^2 + (vy[j] - u0[2])^2 + (vz[k] - u0[3])^2) / vth^2)
        for i in eachindex(vx), j in eachindex(vy), k in eachindex(vz)
    ]
    mom = vdf_moments(f, (vx, vy, vz); species = "O2+")
    @test isapprox(mom.n, n0; rtol = 0.02)
    @test isapprox(mom.V, u0; rtol = 0.05)

    refined_cfg = refine_backtrace_config(cfg, f; dv_kms = 5.0, margin_kms = 10.0)
    @test (refined_cfg.vx_max_kms - refined_cfg.vx_min_kms) < (cfg.vx_max_kms - cfg.vx_min_kms)

    if isfile(data_path("mars_fields_spherical_from_dat.vts"))
        f_tot = load_mhd_fields(; electric_field = :total)
        f_gradpe = load_mhd_fields(; electric_field = :gradpe)
        f_res = load_mhd_fields(; electric_field = :res)
        @test size(f_tot.E) == (3, 120, 72, 144)
        @test all(isfinite, f_tot.E) && all(isfinite, f_gradpe.E) && all(isfinite, f_res.E)
        itp_full = build_field_work_interpolators()
        @test itp_full.gradpe !== nothing && itp_full.res !== nothing
    end
end

