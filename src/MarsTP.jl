module MarsTP

using LinearAlgebra
using Printf
using Random
using Statistics
using StaticArrays
using MAT
using JLD2
using JSON
using TOML
using ReadVTK
using WriteVTK
using PyPlot
if ccall(:jl_generating_output, Cint, ()) == 1 && getfield(PyPlot.matplotlib, :o) == C_NULL
    PyPlot.PyCall.copy!(PyPlot.matplotlib, PyPlot.PyCall.pyimport("matplotlib"))
    using Batsrus
    setfield!(PyPlot.matplotlib, :o, PyPlot.PyCall.PyPtr(C_NULL))
else
    using Batsrus
end
using FastInterpolations
using BorisPushers
using TestParticle
import KernelAbstractions as KA
import NaturalNeighbours as NN
using VelocityDistributionFunctions

import TestParticle as TP
import VelocityDistributionFunctions as VDF

export Rm, Rinner, Router
export project_path, data_path, resolve_project_path
export MHDFields, SpeciesMoments, load_mhd_fields, load_mhd_moments, mhd_param
export load_batsrus_tecplot, convert_batsrus_to_vts, interpolate_batsrus_slice_nn
export load_gitm, load_amps, neutral_properties, hot_oxygen_density
export O2plusSourceRates, load_o2plus_source_rates, assert_same_grid
export calculate_cell_volumes, calculate_surface_areas, compute_o2plus_source_rates
export IonosphereSource, load_ionosphere_source, ionosphere_properties, ionosphere_distribution
export BacktraceConfig, run_backtrace_vdf, velocity_axes
export ForwardTraceConfig, trace_forward, trace_forward_bounded
export detector_omni_def
export forward_psd, ForwardPSDAccumulator, accumulate_forward_psd!, finish_forward_psd
export write_trajectory_batch, foreach_saved_trajectory, forward_psd_saved
export MonteCarloWeight, thermal_speed_from_temperature_ev, maxwellian_importance_weight_3d
export particle_density_weight, sample_maxwellian_source
export trajectory_work, build_field_work_interpolators, field_work
export Electric_field_work_profile, field_work_profile, particle_field_work
export adiabatic_work_profile
export get_work_rates, get_work_channels, get_magnetic_properties, get_adiabaticity
export plot_vx_vz, load_detector_vdf, vdf_moments, refine_backtrace_config
export o2plus_production_density
export MODELS, boundary_profile, plot_mars, bs_mpb, plot_mars_context
export load_trajectories, plot_trajectory

const get_work_rates = TP.get_work_rates
const get_work_channels = TP.get_work_channels
const get_magnetic_properties = TP.get_magnetic_properties
const get_adiabaticity = TP.get_adiabaticity

include("constants.jl")
include("paths.jl")
include("data/mhd.jl")
include("data/atmosphere.jl")
include("data/source_rates.jl")
include("chemistry/reactions.jl")
include("analysis/detector_omni_def.jl")
include("tracing/detector_psd_backward.jl")
include("tracing/forward_tracing.jl")
include("tracing/forward_bounded.jl")
include("tracing/monte_carlo_weight.jl")
include("tracing/detector_psd_forward.jl")
include("tracing/forward_psd_accumulator.jl")
include("tracing/trajectory_io.jl")
include("analysis/electric_field_work.jl")
include("analysis/detector_vdf.jl")
include("visualization/mars.jl")
include("visualization/plot_trajectory.jl")

end
