# Plot a reproducible uniform sample of 5000 propagated trajectories using MarsTP and PyPlot.
using MarsTP
using PyPlot

function main(args = ARGS)
    isempty(args) && error("Usage: julia --project=. plot_trajectories.jl RUN_DIR [OUTPUT_PNG]")
    run_dir = abspath(args[1])
    out_png = length(args) >= 2 ? abspath(args[2]) : joinpath(pwd(), "trajectories_5000.png")
    fig, _, records = plot_trajectory(
        run_dir;
        planes = ("XZ", "XY", "YZ"),
        count = 5000,
        output = out_png,
    )
    PyPlot.plt.close(fig)
    println("Saved $out_png: $(length(records)) trajectories")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

