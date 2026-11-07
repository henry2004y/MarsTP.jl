struct GITMAtmosphere{NO, NCO2, TN, POP, POP2, PCO2P}
    nO::NO
    nCO2::NCO2
    Tn::TN
    PH_Op::POP
    PH_Op2::POP2
    PH_CO2p::PCO2P
end

struct AMPSAtmosphere{NOH}
    nO_hot::NOH
end

function load_gitm(path = data_path("gitm_sph.mat"))
    mat = matread(resolve_project_path(path))
    r = Float64.(vec(mat["r"]))
    theta = reverse(Float64.(vec(mat["theta"])))
    phi = Float64.(vec(mat["phi"]))
    return GITMAtmosphere(
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["nO"]), dims = 2), r, theta, phi),
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["nCO2"]), dims = 2), r, theta, phi),
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["Tn"]), dims = 2), r, theta, phi),
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["PH_Op"]), dims = 2), r, theta, phi),
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["PH_Op2"]), dims = 2), r, theta, phi),
        TP.build_interpolator(TP.StructuredGrid, reverse(Float64.(mat["PH_CO2p"]), dims = 2), r, theta, phi),
    )
end

function load_amps(path = data_path("amps_sph.mat"))
    mat = matread(resolve_project_path(path))
    r = Float64.(vec(mat["r"]))
    theta = reverse(Float64.(vec(mat["theta"])))
    phi = Float64.(vec(mat["phi"]))
    nO_hot = reverse(Float64.(mat["nO_hot"]), dims = 2)
    return AMPSAtmosphere(TP.build_interpolator(TP.StructuredGrid, nO_hot, r, theta, phi))
end

neutral_properties(gitm::GITMAtmosphere, x) = neutral_properties(gitm, TP.cart2sph(x)...)

function neutral_properties(gitm::GITMAtmosphere, r, theta, phi)
    if r - Rm <= GITM_H_THRESHOLD
        x = TP.sph2cart(r, theta, phi)
        return (nO = gitm.nO(x), nCO2 = gitm.nCO2(x), Tn = gitm.Tn(x))
    end

    dh = r - Rm - GITM_H_THRESHOLD
    x220 = TP.sph2cart(Rm + GITM_H_THRESHOLD - 5.0, theta, phi)
    nO0 = gitm.nO(x220)
    nCO20 = gitm.nCO2(x220)
    Tn0 = gitm.Tn(x220)
    HO = TP.kB * Tn0 / (m_O * g_mars)
    HCO2 = TP.kB * Tn0 / (m_CO2 * g_mars)
    nO = dh / HO > 20.0 ? 0.0 : nO0 * exp(-dh / HO)
    nCO2 = dh / HCO2 > 20.0 ? 0.0 : nCO20 * exp(-dh / HCO2)
    return (nO = nO, nCO2 = nCO2, Tn = Tn0)
end

@inline function _neutral_temperature(gitm::GITMAtmosphere, x)
    r, theta, phi = TP.cart2sph(x)
    if r - Rm <= GITM_H_THRESHOLD
        return gitm.Tn(TP.sph2cart(r, theta, phi))
    end
    return gitm.Tn(TP.sph2cart(Rm + GITM_H_THRESHOLD - 5.0, theta, phi))
end

function hot_oxygen_density(amps::AMPSAtmosphere, x)
    norm(x) - Rm <= 100.0e3 && return 0.0
    return amps.nO_hot(x)
end

function hot_oxygen_density(amps::AMPSAtmosphere, r, theta, phi)
    r - Rm <= 100.0e3 && return 0.0
    return amps.nO_hot(TP.sph2cart(r, theta, phi))
end
