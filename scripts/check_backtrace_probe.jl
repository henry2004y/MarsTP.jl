using MarsTP, StaticArrays, LinearAlgebra
fields = load_mhd_fields()
p = SA[0.0, 0.0, 2Rm]
param = MarsTP.mhd_param(fields; species="O2+")
E, B = param.E(p, 0.0), param.B(p, 0.0)
println("Probe [Rm]: ", p / Rm)
println("E [V/m]: ", E, "; B [T]: ", B)
@assert all(isfinite, E) && all(isfinite, B)
println("Field domain [Rm]: ", extrema(fields.r) ./ Rm)
println("TestParticle source: ", pathof(MarsTP.TP))
