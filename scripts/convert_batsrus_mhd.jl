using MarsTP
using Printf

function parse_args(args)
    opts = Dict{String, String}(
        "with_crustal_dat" => project_path("BATSRUS-data", "MHD-with-crustal", "3d__var_1_n00150000.dat"),
        "without_crustal_dat" => project_path("BATSRUS-data", "MHD-without-crustal", "3d__var_1_n00120000.dat"),
        "out_with_crustal" => data_path("mars_fields_with_crustal.vts"),
        "out_without_crustal" => data_path("mars_fields_without_crustal.vts"),
        "out_default" => data_path("mars_fields_spherical_from_dat.vts"),
        "default_case" => "with_crustal",
        "compute_source_rates" => "true",
    )
    for arg in args
        startswith(arg, "--") || continue
        parts = split(arg[3:end], "="; limit = 2)
        length(parts) == 2 && (opts[parts[1]] = parts[2])
    end
    return opts
end

function main(args = ARGS)
    opts = parse_args(args)
    do_sources = lowercase(opts["compute_source_rates"]) in ("true", "1", "yes")

    cases = (
        ("with_crustal", opts["with_crustal_dat"], opts["out_with_crustal"], data_path("O2plus_source_rates_with_crustal.mat")),
        ("without_crustal", opts["without_crustal_dat"], opts["out_without_crustal"], data_path("O2plus_source_rates_without_crustal.mat")),
    )

    for (label, dat_path, vts_path, src_mat_path) in cases
        if !isfile(dat_path)
            @printf("Skipping %s (input not found: %s)\n", label, dat_path)
            continue
        end
        t0 = time()
        @printf("Converting %s -> %s ...\n", dat_path, vts_path)
        saved_vts = convert_batsrus_to_vts(dat_path, vts_path)
        @printf("  Wrote %s in %.2f s\n", saved_vts, time() - t0)

        fields = load_mhd_fields(saved_vts; electric_field = :total)
        moments = load_mhd_moments(saved_vts)
        @printf("  Verified VTS grid: (%d, %d, %d), |B| max = %.3e T, |E_total| max = %.3e V/m, n(O2+) max = %.3e m^-3\n",
            length(fields.r), length(fields.theta), length(fields.phi),
            maximum(abs, fields.B), maximum(abs, fields.E), maximum(moments.O2plus.n))

        if do_sources && isfile(data_path("gitm_sph.mat")) && isfile(data_path("amps_sph.mat"))
            t_src = time()
            sr = compute_o2plus_source_rates(saved_vts; output_path = src_mat_path)
            assert_same_grid(sr, fields)
            @printf("  Wrote %s in %.2f s (total production = %.3e s^-1, total outflow = %.3e s^-1)\n",
                src_mat_path, time() - t_src, sum(sr.production_rate), sum(sr.outflow_rate))
        end

        if opts["default_case"] == label
            cp(saved_vts, opts["out_default"]; force = true)
            @printf("  Updated default VTS %s -> %s\n", opts["out_default"], label)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

