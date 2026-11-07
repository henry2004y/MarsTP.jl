const TEMP_E = [
    1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4, 7.0e4, 8.0e4, 9.0e4,
    1.0e5, 1.5e5, 2.0e5, 3.0e5, 4.0e5, 5.0e5, 6.0e5, 7.0e5,
    8.0e5, 9.0e5, 1.0e6, 1.5e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6,
    7.0e6, 1.0e7,
]

const CM3S_TO_M3S = 1.0e-6
const RATE_EI_O = [
    4.682e-16, 2.172e-12, 4.547e-11, 2.288e-10, 6.318e-10, 1.275e-9,
    2.14e-9, 3.188e-9, 4.36e-9, 5.684e-9, 1.29e-8, 2.008e-8,
    3.241e-8, 4.201e-8, 4.955e-8, 5.554e-8, 6.047e-8, 6.452e-8,
    6.791e-8, 7.078e-8, 8.006e-8, 8.466e-8, 8.789e-8, 8.774e-8,
    8.625e-8, 8.257e-8, 7.709e-8,
] .* CM3S_TO_M3S

const RATE_EI_O_ITP = linear_interp(TEMP_E, RATE_EI_O; extrap = ClampExtrap())

CO2p_O_to_O2p_CO(nCO2p, nO) = @. (1.64e-10 * CM3S_TO_M3S) * nCO2p * nO
CO2_Op_to_O2p_CO(nCO2, nOp, Ti) = @. (1.1e-9 * (800.0 / Ti)^0.39 * CM3S_TO_M3S) * nCO2 * nOp

function o2plus_production_density(moments, gitm::GITMAtmosphere, amps::AMPSAtmosphere, r, theta, phi)
    Ti = @. (moments.Hplus.t + moments.Oplus.t + moments.O2plus.t + moments.CO2plus.t) / 4
    r3 = reshape(r, :, 1, 1)
    t3 = reshape(theta, 1, :, 1)
    p3 = reshape(phi, 1, 1, :)

    neutral = neutral_properties.(Ref(gitm), r3, t3, p3)
    nO_cold = getproperty.(neutral, :nO)
    nCO2 = getproperty.(neutral, :nCO2)
    nO_hot = hot_oxygen_density.(Ref(amps), r3, t3, p3)
    nO = nO_cold .+ nO_hot

    return (
        CO2p_O2p = CO2p_O_to_O2p_CO(moments.CO2plus.n, nO),
        Op_O2p = CO2_Op_to_O2p_CO(nCO2, moments.Oplus.n, Ti),
    )
end
