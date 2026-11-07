struct MHDFields
    r::Vector{Float64}
    theta::Vector{Float64}
    phi::Vector{Float64}
    E::Array{Float64, 4}
    B::Array{Float64, 4}
    electric_field::Symbol
    path::String
end

struct SpeciesMoments
    n::Array{Float64, 3}
    v::Array{Float64, 4}
    t::Array{Float64, 3}
end

const MU0_SI = 4.0e-7 * pi

function _mhd_axes(nr, ntheta, nphi; r_inner = Rinner, r_outer = Router)
    return (
        collect(exp.(range(log(Float64(r_inner)), log(Float64(r_outer)), length = nr))),
        collect(range(0.0, pi, length = ntheta)),
        collect(range(0.0, 2pi, length = nphi)),
    )
end

function _mhd_axes(vtk, nr, ntheta, nphi)
    grid = ReadVTK.LightXML.root(vtk.xml_file)[vtk.file_type][1]
    isempty(grid["FieldData"]) && return _mhd_axes(nr, ntheta, nphi)
    metadata = get_field_data(vtk)
    names = ("r_axis [m]", "theta_axis [rad]", "phi_axis [rad]")
    present = map(name -> name in keys(metadata), names)
    any(present) || return _mhd_axes(nr, ntheta, nphi)
    all(present) || throw(ArgumentError("Incomplete MHD grid-axis metadata"))
    axes = map(name -> Float64.(vec(get_data(metadata[name]))), names)
    for (axis, n) in zip(axes, (nr, ntheta, nphi))
        length(axis) == n && all(isfinite, axis) && all(>(0), diff(axis)) ||
            throw(ArgumentError("Invalid MHD grid-axis metadata"))
    end
    first(axes[1]) > 0 || throw(ArgumentError("MHD radii must be positive"))
    if "Planet_radius [m]" in keys(metadata)
        radius = only(get_data(metadata["Planet_radius [m]"]))
        isapprox(radius, Rm; rtol = 1e-12) || throw(ArgumentError("MHD planet radius does not match MarsTP Rm"))
    end
    return axes
end

function _point_vector(point_data, names; scale = 1.0)
    keys_available = keys(point_data)
    for name in names
        if name in keys_available
            data = get_data(point_data[name])
            return scale == 1.0 ? data : data .* scale
        end
    end
    error("Missing vector field. Tried: $(join(names, ", "))")
end

function _electric_names(which)
    which in (:total, "total") && return ("E_Total [V/m]", "E_total [V/m]", "E [V/m]")
    which in (:conv, :convection, "conv", "convection") && return ("E_conv [V/m]", "Econv")
    which in (:hall, "hall") && return ("E_hall [V/m]", "Ehall")
    which in (:gradpe, "gradpe") && return ("E_gradpe [V/m]", "Egradpe")
    which in (:res, :resistive, "res", "resistive") && return ("E_res [V/m]", "Eres")
    which in (:plot, "plot") && return ("E_plot [V/m]", "Eplot")
    which isa AbstractString && return (which,)
    error("Unknown electric field selector: $which")
end

function _rotate_vectors_to_spherical!(A, r, theta, phi)
    for k in eachindex(phi), j in eachindex(theta)
        sin_theta, cos_theta = sincos(theta[j])
        sin_phi, cos_phi = sincos(phi[k])
        for i in eachindex(r)
            ax, ay, az = A[1, i, j, k], A[2, i, j, k], A[3, i, j, k]
            A[1, i, j, k] = ax * sin_theta * cos_phi + ay * sin_theta * sin_phi + az * cos_theta
            A[2, i, j, k] = ax * cos_theta * cos_phi + ay * cos_theta * sin_phi - az * sin_theta
            A[3, i, j, k] = -ax * sin_phi + ay * cos_phi
        end
    end
    return A
end

function _rotate_vectors_to_cartesian!(A, r, theta, phi)
    for k in eachindex(phi), j in eachindex(theta), i in eachindex(r)
        ax, ay, az = TP.sph2cartvec(A[1, i, j, k], A[2, i, j, k], A[3, i, j, k], theta[j], phi[k])
        A[1, i, j, k] = ax
        A[2, i, j, k] = ay
        A[3, i, j, k] = az
    end
    return A
end

function _find_var_index(variables, candidates; required::Bool = true)
    cleaned = [ lowercase(strip(String(v))) for v in variables ]
    for cand in candidates
        c = lowercase(cand)
        idx = findfirst(v -> v == c || startswith(v, c * " ") || startswith(v, c * "[") || startswith(v, c * "_"), cleaned)
        idx !== nothing && return idx
    end
    required && error("Variable not found in BATSRUS file. Tried: $(join(candidates, ", ")); available: $(join(variables, ", "))")
    return nothing
end

function _nearest_sorted_index(sorted_vec::Vector{Float64}, val::Float64)
    n = length(sorted_vec)
    idx = searchsortedfirst(sorted_vec, val)
    idx <= 1 && return 1
    idx > n && return n
    return abs(val - sorted_vec[idx - 1]) <= abs(sorted_vec[idx] - val) ? (idx - 1) : idx
end

function _repair_pole_slice!(field_slice::AbstractMatrix{Float64}, theta::Vector{Float64}; positive::Bool = false)
    nr, ntheta = size(field_slice, 1), length(theta)
    nphi = size(field_slice, 2)
    # Periodic mean over unique phi indices 1:(nphi-1)
    nph_u = max(nphi - 1, 1)
    for (pole_j, r1, r2, r3) in ((1, 2, 3, 4), (ntheta, ntheta - 1, ntheta - 2, ntheta - 3))
        dth1 = (theta[r1] - theta[pole_j])^2
        dth2 = (theta[r2] - theta[pole_j])^2
        dth3 = (theta[r3] - theta[pole_j])^2
        # Quadratic fit in dth = (theta - theta_pole)^2 at 3 rings -> intercept at dth = 0
        # Lagrange basis at 0 for nodes (dth1, dth2, dth3):
        L1 = (dth2 * dth3) / ((dth1 - dth2) * (dth1 - dth3))
        L2 = (dth1 * dth3) / ((dth2 - dth1) * (dth2 - dth3))
        L3 = (dth1 * dth2) / ((dth3 - dth1) * (dth3 - dth2))
        for ir in 1:nr
            m1 = sum(@view field_slice[ir, r1, 1:nph_u]) / nph_u
            m2 = sum(@view field_slice[ir, r2, 1:nph_u]) / nph_u
            m3 = sum(@view field_slice[ir, r3, 1:nph_u]) / nph_u
            ring_est = L1 * m1 + L2 * m2 + L3 * m3
            if positive && (ring_est <= 0.0 || abs(ring_est - m1) > 0.5 * abs(m1))
                ring_est = max(m1, eps(Float64))
            end
            pole_val = field_slice[ir, pole_j, 1]
            # Check if native pole value is finite and agrees with nearest ring within 25%
            valid_native = isfinite(pole_val) && (!positive || pole_val > 0.0) &&
                (abs(m1) <= 1e-30 || abs(pole_val - m1) <= 0.25 * max(abs(m1), abs(ring_est)))
            chosen = valid_native ? pole_val : ring_est
            field_slice[ir, pole_j, :] .= chosen
        end
    end
    return field_slice
end

function _repair_poles_3d!(A::Array{Float64, 3}, theta::Vector{Float64}; positive::Bool = false)
    nr, ntheta, nphi = size(A)
    nph_u = max(nphi - 1, 1)
    for (pole_j, r1, r2, r3) in ((1, 2, 3, 4), (ntheta, ntheta - 1, ntheta - 2, ntheta - 3))
        dth1 = (theta[r1] - theta[pole_j])^2
        dth2 = (theta[r2] - theta[pole_j])^2
        dth3 = (theta[r3] - theta[pole_j])^2
        L1 = (dth2 * dth3) / ((dth1 - dth2) * (dth1 - dth3))
        L2 = (dth1 * dth3) / ((dth2 - dth1) * (dth2 - dth3))
        L3 = (dth1 * dth2) / ((dth3 - dth1) * (dth3 - dth2))
        for ir in 1:nr
            m1 = sum(@view A[ir, r1, 1:nph_u]) / nph_u
            m2 = sum(@view A[ir, r2, 1:nph_u]) / nph_u
            m3 = sum(@view A[ir, r3, 1:nph_u]) / nph_u
            ring_est = L1 * m1 + L2 * m2 + L3 * m3
            if positive && (ring_est <= 0.0 || abs(ring_est - m1) > 0.5 * abs(m1))
                ring_est = max(m1, eps(Float64))
            end
            pole_val = A[ir, pole_j, 1]
            valid_native = isfinite(pole_val) && (!positive || pole_val > 0.0) &&
                (abs(m1) <= 1e-30 || abs(pole_val - m1) <= 0.25 * max(abs(m1), abs(ring_est)))
            chosen = valid_native ? pole_val : ring_est
            A[ir, pole_j, :] .= chosen
        end
    end
    return A
end

function _repair_poles_4d!(A::Array{Float64, 4}, theta::Vector{Float64})
    for comp in 1:size(A, 1)
        slice = A[comp, :, :, :]
        _repair_poles_3d!(slice, theta; positive = false)
        A[comp, :, :, :] .= slice
    end
    return A
end

"""
    load_batsrus_tecplot(dat_path; polar_repair=true)

Load a 3D BATSRUS ASCII Tecplot `.dat` file using `Batsrus.readtecdata`, assemble its
native spherical grid `(r, theta, phi)`, reconstruct all species moments (`Hplus`,
`Hswp`, `Hpp`, `Oplus`, `O2plus`, `CO2plus`, `Te`, `ne`) and electric field channels
(`E_total`, `E_conv`, `E_hall`, `E_gradpe`, `E_res`, `E_plot`), and return a NamedTuple.
"""
function load_batsrus_tecplot(dat_path::AbstractString; polar_repair::Bool = true)
    resolved = resolve_project_path(dat_path)
    hdr, data, _ = Batsrus.readtecdata(resolved)
    vars = hdr.variable

    ix_x = _find_var_index(vars, ("x",))
    ix_y = _find_var_index(vars, ("y",))
    ix_z = _find_var_index(vars, ("z",))
    ix_hswp = _find_var_index(vars, ("hswp",))
    ix_hpp = _find_var_index(vars, ("hpp", "hp"))
    ix_o2p = _find_var_index(vars, ("o2p",))
    ix_op = _find_var_index(vars, ("op",))
    ix_co2p = _find_var_index(vars, ("co2p",))
    ix_ux = _find_var_index(vars, ("u_x", "ux"))
    ix_uy = _find_var_index(vars, ("u_y", "uy"))
    ix_uz = _find_var_index(vars, ("u_z", "uz"))
    ix_bx = _find_var_index(vars, ("b_x", "bx"))
    ix_by = _find_var_index(vars, ("b_y", "by"))
    ix_bz = _find_var_index(vars, ("b_z", "bz"))
    ix_pe = _find_var_index(vars, ("pe", "p_e"))
    ix_p = _find_var_index(vars, ("p",))
    ix_jx = _find_var_index(vars, ("j_x", "jx"))
    ix_jy = _find_var_index(vars, ("j_y", "jy"))
    ix_jz = _find_var_index(vars, ("j_z", "jz"))
    ix_eta = _find_var_index(vars, ("eta",); required = false)
    ix_gpx = _find_var_index(vars, ("gradpe_x",); required = false)
    ix_gpy = _find_var_index(vars, ("gradpe_y",); required = false)
    ix_gpz = _find_var_index(vars, ("gradpe_z",); required = false)
    ix_ex = _find_var_index(vars, ("e_x", "ex"); required = false)
    ix_ey = _find_var_index(vars, ("e_y", "ey"); required = false)
    ix_ez = _find_var_index(vars, ("e_z", "ez"); required = false)

    x_R = @view data[ix_x, :]
    y_R = @view data[ix_y, :]
    z_R = @view data[ix_z, :]

    pole_mask = (x_R .== 0.0f0) .& (y_R .== 0.0f0)
    r_R = sort(unique(Float64.(abs.(z_R[pole_mask]))))
    nr = length(r_R)
    ntheta = 61
    nphi_unique = 120
    nphi = nphi_unique + 1
    dtheta = pi / (ntheta - 1)
    dphi = 2pi / nphi_unique

    r_m = r_R .* Rm
    theta = collect(range(0.0, pi, length = ntheta))
    phi = collect(range(0.0, 2pi, length = nphi))

    n_Hswp = zeros(Float64, nr, ntheta, nphi)
    n_Hpp = zeros(Float64, nr, ntheta, nphi)
    n_Op = zeros(Float64, nr, ntheta, nphi)
    n_O2p = zeros(Float64, nr, ntheta, nphi)
    n_CO2p = zeros(Float64, nr, ntheta, nphi)
    pe_Pa = zeros(Float64, nr, ntheta, nphi)
    P_Pa = zeros(Float64, nr, ntheta, nphi)
    eta_grid = zeros(Float64, nr, ntheta, nphi)

    U_ms = zeros(Float64, 3, nr, ntheta, nphi)
    B_T = zeros(Float64, 3, nr, ntheta, nphi)
    J_Am2 = zeros(Float64, 3, nr, ntheta, nphi)
    gradpe_Pam = zeros(Float64, 3, nr, ntheta, nphi)
    E_plot = zeros(Float64, 3, nr, ntheta, nphi)
    pole_counts = zeros(Int, nr, 2)

    n_nodes = size(data, 2)
    for k in 1:n_nodes
        xk = Float64(x_R[k])
        yk = Float64(y_R[k])
        zk = Float64(z_R[k])
        rk = hypot(xk, yk, zk)
        ir = _nearest_sorted_index(r_R, rk)
        thk = acos(clamp(zk / rk, -1.0, 1.0))
        ith = clamp(round(Int, thk / dtheta) + 1, 1, ntheta)

        # In BATSRUS Mars multi-species output, Rho [amu/cm^3] = Hswp + Hpp + 32*O2p + 16*Op + 44*Co2p,
        # so Hswp, Hpp, O2p, Op, Co2p are already number densities in cm^-3.
        hswp_val = max(Float64(data[ix_hswp, k]), 0.0) * 1.0e6
        hpp_val = max(Float64(data[ix_hpp, k]), 0.0) * 1.0e6
        o2p_val = max(Float64(data[ix_o2p, k]), 0.0) * 1.0e6
        op_val = max(Float64(data[ix_op, k]), 0.0) * 1.0e6
        co2p_val = max(Float64(data[ix_co2p, k]), 0.0) * 1.0e6
        pe_val = max(Float64(data[ix_pe, k]), 0.0) * 1.0e-9
        p_val = max(Float64(data[ix_p, k]), 0.0) * 1.0e-9
        eta_val = ix_eta === nothing ? 0.0 : Float64(data[ix_eta, k])

        ux_val = Float64(data[ix_ux, k]) * 1.0e3
        uy_val = Float64(data[ix_uy, k]) * 1.0e3
        uz_val = Float64(data[ix_uz, k]) * 1.0e3

        bx_val = Float64(data[ix_bx, k]) * 1.0e-9
        by_val = Float64(data[ix_by, k]) * 1.0e-9
        bz_val = Float64(data[ix_bz, k]) * 1.0e-9

        jx_val = Float64(data[ix_jx, k]) * 1.0e-6
        jy_val = Float64(data[ix_jy, k]) * 1.0e-6
        jz_val = Float64(data[ix_jz, k]) * 1.0e-6

        gpx_val = ix_gpx === nothing ? 0.0 : Float64(data[ix_gpx, k]) * 1.0e-9
        gpy_val = ix_gpy === nothing ? 0.0 : Float64(data[ix_gpy, k]) * 1.0e-9
        gpz_val = ix_gpz === nothing ? 0.0 : Float64(data[ix_gpz, k]) * 1.0e-9

        ex_val = ix_ex === nothing ? 0.0 : Float64(data[ix_ex, k]) * 1.0e-3
        ey_val = ix_ey === nothing ? 0.0 : Float64(data[ix_ey, k]) * 1.0e-3
        ez_val = ix_ez === nothing ? 0.0 : Float64(data[ix_ez, k]) * 1.0e-3

        if ith == 1 || ith == ntheta
            pidx = ith == 1 ? 1 : 2
            c = pole_counts[ir, pidx] + 1
            pole_counts[ir, pidx] = c
            w = 1.0 / c
            n_Hswp[ir, ith, 1] += (hswp_val - n_Hswp[ir, ith, 1]) * w
            n_Hpp[ir, ith, 1] += (hpp_val - n_Hpp[ir, ith, 1]) * w
            n_O2p[ir, ith, 1] += (o2p_val - n_O2p[ir, ith, 1]) * w
            n_Op[ir, ith, 1] += (op_val - n_Op[ir, ith, 1]) * w
            n_CO2p[ir, ith, 1] += (co2p_val - n_CO2p[ir, ith, 1]) * w
            pe_Pa[ir, ith, 1] += (pe_val - pe_Pa[ir, ith, 1]) * w
            P_Pa[ir, ith, 1] += (p_val - P_Pa[ir, ith, 1]) * w
            eta_grid[ir, ith, 1] += (eta_val - eta_grid[ir, ith, 1]) * w

            U_ms[1, ir, ith, 1] += (ux_val - U_ms[1, ir, ith, 1]) * w
            U_ms[2, ir, ith, 1] += (uy_val - U_ms[2, ir, ith, 1]) * w
            U_ms[3, ir, ith, 1] += (uz_val - U_ms[3, ir, ith, 1]) * w

            B_T[1, ir, ith, 1] += (bx_val - B_T[1, ir, ith, 1]) * w
            B_T[2, ir, ith, 1] += (by_val - B_T[2, ir, ith, 1]) * w
            B_T[3, ir, ith, 1] += (bz_val - B_T[3, ir, ith, 1]) * w

            J_Am2[1, ir, ith, 1] += (jx_val - J_Am2[1, ir, ith, 1]) * w
            J_Am2[2, ir, ith, 1] += (jy_val - J_Am2[2, ir, ith, 1]) * w
            J_Am2[3, ir, ith, 1] += (jz_val - J_Am2[3, ir, ith, 1]) * w

            gradpe_Pam[1, ir, ith, 1] += (gpx_val - gradpe_Pam[1, ir, ith, 1]) * w
            gradpe_Pam[2, ir, ith, 1] += (gpy_val - gradpe_Pam[2, ir, ith, 1]) * w
            gradpe_Pam[3, ir, ith, 1] += (gpz_val - gradpe_Pam[3, ir, ith, 1]) * w

            E_plot[1, ir, ith, 1] += (ex_val - E_plot[1, ir, ith, 1]) * w
            E_plot[2, ir, ith, 1] += (ey_val - E_plot[2, ir, ith, 1]) * w
            E_plot[3, ir, ith, 1] += (ez_val - E_plot[3, ir, ith, 1]) * w
        else
            phk = mod2pi(atan(yk, xk))
            iph = mod(round(Int, phk / dphi), nphi_unique) + 1
            n_Hswp[ir, ith, iph] = hswp_val
            n_Hpp[ir, ith, iph] = hpp_val
            n_O2p[ir, ith, iph] = o2p_val
            n_Op[ir, ith, iph] = op_val
            n_CO2p[ir, ith, iph] = co2p_val
            pe_Pa[ir, ith, iph] = pe_val
            P_Pa[ir, ith, iph] = p_val
            eta_grid[ir, ith, iph] = eta_val

            U_ms[1, ir, ith, iph] = ux_val
            U_ms[2, ir, ith, iph] = uy_val
            U_ms[3, ir, ith, iph] = uz_val

            B_T[1, ir, ith, iph] = bx_val
            B_T[2, ir, ith, iph] = by_val
            B_T[3, ir, ith, iph] = bz_val

            J_Am2[1, ir, ith, iph] = jx_val
            J_Am2[2, ir, ith, iph] = jy_val
            J_Am2[3, ir, ith, iph] = jz_val

            gradpe_Pam[1, ir, ith, iph] = gpx_val
            gradpe_Pam[2, ir, ith, iph] = gpy_val
            gradpe_Pam[3, ir, ith, iph] = gpz_val

            E_plot[1, ir, ith, iph] = ex_val
            E_plot[2, ir, ith, iph] = ey_val
            E_plot[3, ir, ith, iph] = ez_val
        end
    end

    # Broadcast pole value across all phi and wrap periodic seam phi[end] = phi[1]
    for scalar_arr in (n_Hswp, n_Hpp, n_O2p, n_Op, n_CO2p, pe_Pa, P_Pa, eta_grid)
        for ith in (1, ntheta), iph in 2:nphi
            scalar_arr[:, ith, iph] .= @view scalar_arr[:, ith, 1]
        end
        scalar_arr[:, :, nphi] .= @view scalar_arr[:, :, 1]
    end
    for vec_arr in (U_ms, B_T, J_Am2, gradpe_Pam, E_plot)
        for ith in (1, ntheta), iph in 2:nphi
            vec_arr[:, :, ith, iph] .= @view vec_arr[:, :, ith, 1]
        end
        vec_arr[:, :, :, nphi] .= @view vec_arr[:, :, :, 1]
    end

    if polar_repair
        for pos_arr in (n_Hswp, n_Hpp, n_O2p, n_Op, n_CO2p, pe_Pa, P_Pa)
            _repair_poles_3d!(pos_arr, theta; positive = true)
        end
        _repair_poles_3d!(eta_grid, theta; positive = false)
        for vec_arr in (U_ms, B_T, J_Am2, gradpe_Pam, E_plot)
            _repair_poles_4d!(vec_arr, theta)
        end
    end

    n_Hp = n_Hswp .+ n_Hpp
    n_e = max.(n_Hp .+ n_Op .+ n_O2p .+ n_CO2p, 1.0e-12)
    T_e = pe_Pa ./ (n_e .* TP.kB)
    T_i = P_Pa ./ (n_e .* TP.kB)

    E_conv = zeros(Float64, 3, nr, ntheta, nphi)
    E_hall = zeros(Float64, 3, nr, ntheta, nphi)
    E_gradpe = zeros(Float64, 3, nr, ntheta, nphi)
    E_res = zeros(Float64, 3, nr, ntheta, nphi)
    E_total = zeros(Float64, 3, nr, ntheta, nphi)

    q_e = TP.qᵢ
    for k in 1:nphi, j in 1:ntheta, i in 1:nr
        ux, uy, uz = U_ms[1, i, j, k], U_ms[2, i, j, k], U_ms[3, i, j, k]
        bx, by, bz = B_T[1, i, j, k], B_T[2, i, j, k], B_T[3, i, j, k]
        jx, jy, jz = J_Am2[1, i, j, k], J_Am2[2, i, j, k], J_Am2[3, i, j, k]
        gpx, gpy, gpz = gradpe_Pam[1, i, j, k], gradpe_Pam[2, i, j, k], gradpe_Pam[3, i, j, k]
        ne_ijk = n_e[i, j, k]
        inv_nee = 1.0 / (ne_ijk * q_e)
        eta_ijk = eta_grid[i, j, k]

        # E_conv = -U x B
        ecx = -(uy * bz - uz * by)
        ecy = -(uz * bx - ux * bz)
        ecz = -(ux * by - uy * bx)

        # E_hall = (J x B) / (n_e e)
        ehx = (jy * bz - jz * by) * inv_nee
        ehy = (jz * bx - jx * bz) * inv_nee
        ehz = (jx * by - jy * bx) * inv_nee

        # E_gradpe = -GradPe / (n_e e)
        egx = -gpx * inv_nee
        egy = -gpy * inv_nee
        egz = -gpz * inv_nee

        # E_res = eta * mu0 * J
        erx = eta_ijk * MU0_SI * jx
        ery = eta_ijk * MU0_SI * jy
        erz = eta_ijk * MU0_SI * jz

        E_conv[1, i, j, k] = ecx
        E_conv[2, i, j, k] = ecy
        E_conv[3, i, j, k] = ecz

        E_hall[1, i, j, k] = ehx
        E_hall[2, i, j, k] = ehy
        E_hall[3, i, j, k] = ehz

        E_gradpe[1, i, j, k] = egx
        E_gradpe[2, i, j, k] = egy
        E_gradpe[3, i, j, k] = egz

        E_res[1, i, j, k] = erx
        E_res[2, i, j, k] = ery
        E_res[3, i, j, k] = erz

        E_total[1, i, j, k] = ecx + ehx + egx + erx
        E_total[2, i, j, k] = ecy + ehy + egy + ery
        E_total[3, i, j, k] = ecz + ehz + egz + erz
    end

    return (;
        path = resolved,
        header = hdr,
        r = r_m,
        theta = theta,
        phi = phi,
        B_T = B_T,
        E_total = E_total,
        E_conv = E_conv,
        E_hall = E_hall,
        E_gradpe = E_gradpe,
        E_res = E_res,
        E_plot = E_plot,
        n_Hp = n_Hp,
        n_Hswp = n_Hswp,
        n_Hpp = n_Hpp,
        n_Op = n_Op,
        n_O2p = n_O2p,
        n_CO2p = n_CO2p,
        n_e = n_e,
        U_ms = U_ms,
        T_i = T_i,
        T_e = T_e,
    )
end

function _axis_weights(src_axis::Vector{Float64}, dst_axis::Vector{Float64})
    n_src = length(src_axis)
    n_dst = length(dst_axis)
    idx_lo = Vector{Int}(undef, n_dst)
    idx_hi = Vector{Int}(undef, n_dst)
    w_hi = Vector{Float64}(undef, n_dst)
    for i in 1:n_dst
        x = dst_axis[i]
        if x <= src_axis[1]
            idx_lo[i] = 1
            idx_hi[i] = 1
            w_hi[i] = 0.0
        elseif x >= src_axis[end]
            idx_lo[i] = n_src
            idx_hi[i] = n_src
            w_hi[i] = 0.0
        else
            hi = searchsortedfirst(src_axis, x)
            lo = hi - 1
            idx_lo[i] = lo
            idx_hi[i] = hi
            denom = src_axis[hi] - src_axis[lo]
            w_hi[i] = denom > 0.0 ? (x - src_axis[lo]) / denom : 0.0
        end
    end
    return idx_lo, idx_hi, w_hi
end

function _trilinear_resample_3d(
        A::Array{Float64, 3},
        r_src::Vector{Float64}, th_src::Vector{Float64}, ph_src::Vector{Float64},
        r_dst::Vector{Float64}, th_dst::Vector{Float64}, ph_dst::Vector{Float64},
    )
    ir0, ir1, wr = _axis_weights(r_src, r_dst)
    it0, it1, wt = _axis_weights(th_src, th_dst)
    ip0, ip1, wp = _axis_weights(ph_src, ph_dst)
    nr, nt, np = length(r_dst), length(th_dst), length(ph_dst)
    out = Array{Float64, 3}(undef, nr, nt, np)
    for k in 1:np
        k0, k1, wk = ip0[k], ip1[k], wp[k]
        wk0 = 1.0 - wk
        for j in 1:nt
            j0, j1, wj = it0[j], it1[j], wt[j]
            wj0 = 1.0 - wj
            for i in 1:nr
                i0, i1, wi = ir0[i], ir1[i], wr[i]
                wi0 = 1.0 - wi
                c00 = wi0 * A[i0, j0, k0] + wi * A[i1, j0, k0]
                c10 = wi0 * A[i0, j1, k0] + wi * A[i1, j1, k0]
                c01 = wi0 * A[i0, j0, k1] + wi * A[i1, j0, k1]
                c11 = wi0 * A[i0, j1, k1] + wi * A[i1, j1, k1]
                c0 = wj0 * c00 + wj * c10
                c1 = wj0 * c01 + wj * c11
                out[i, j, k] = wk0 * c0 + wk * c1
            end
        end
    end
    return out
end

function _trilinear_resample_4d(
        A::Array{Float64, 4},
        r_src::Vector{Float64}, th_src::Vector{Float64}, ph_src::Vector{Float64},
        r_dst::Vector{Float64}, th_dst::Vector{Float64}, ph_dst::Vector{Float64},
    )
    ir0, ir1, wr = _axis_weights(r_src, r_dst)
    it0, it1, wt = _axis_weights(th_src, th_dst)
    ip0, ip1, wp = _axis_weights(ph_src, ph_dst)
    nc = size(A, 1)
    nr, nt, np = length(r_dst), length(th_dst), length(ph_dst)
    out = Array{Float64, 4}(undef, nc, nr, nt, np)
    for k in 1:np
        k0, k1, wk = ip0[k], ip1[k], wp[k]
        wk0 = 1.0 - wk
        for j in 1:nt
            j0, j1, wj = it0[j], it1[j], wt[j]
            wj0 = 1.0 - wj
            for i in 1:nr
                i0, i1, wi = ir0[i], ir1[i], wr[i]
                wi0 = 1.0 - wi
                for c in 1:nc
                    c00 = wi0 * A[c, i0, j0, k0] + wi * A[c, i1, j0, k0]
                    c10 = wi0 * A[c, i0, j1, k0] + wi * A[c, i1, j1, k0]
                    c01 = wi0 * A[c, i0, j0, k1] + wi * A[c, i1, j0, k1]
                    c11 = wi0 * A[c, i0, j1, k1] + wi * A[c, i1, j1, k1]
                    c0 = wj0 * c00 + wj * c10
                    c1 = wj0 * c01 + wj * c11
                    out[c, i, j, k] = wk0 * c0 + wk * c1
                end
            end
        end
    end
    return out
end

function _resample_batsrus_grid(
        nat;
        nr::Int = 120,
        ntheta::Int = 72,
        nphi::Int = 144,
        r_inner::Real = Rinner,
        r_outer::Real = Router,
    )
    r, theta, phi = _mhd_axes(nr, ntheta, nphi; r_inner = r_inner, r_outer = r_outer)
    s3(A) = _trilinear_resample_3d(A, nat.r, nat.theta, nat.phi, r, theta, phi)
    s4(A) = _trilinear_resample_4d(A, nat.r, nat.theta, nat.phi, r, theta, phi)

    return (;
        path = nat.path,
        r = r,
        theta = theta,
        phi = phi,
        B_T = s4(nat.B_T),
        E_total = s4(nat.E_total),
        E_conv = s4(nat.E_conv),
        E_hall = s4(nat.E_hall),
        E_gradpe = s4(nat.E_gradpe),
        E_res = s4(nat.E_res),
        E_plot = s4(nat.E_plot),
        n_Hp = s3(nat.n_Hp),
        n_Hswp = s3(nat.n_Hswp),
        n_Hpp = s3(nat.n_Hpp),
        n_Op = s3(nat.n_Op),
        n_O2p = s3(nat.n_O2p),
        n_CO2p = s3(nat.n_CO2p),
        n_e = s3(nat.n_e),
        U_ms = s4(nat.U_ms),
        T_i = s3(nat.T_i),
        T_e = s3(nat.T_e),
    )
end

"""
    convert_batsrus_to_vts(dat_path, out_vts_path; nr=120, ntheta=72, nphi=144,
                           r_inner=Rinner, r_outer=Router, polar_repair=true,
                           compress=false)

Convert a BATSRUS 3D Tecplot `.dat` file into a structured spherical VTK (`.vts`) file
matching the `MarsTP` `load_mhd_fields` and `load_mhd_moments` schema.
"""
function convert_batsrus_to_vts(
        dat_path::AbstractString,
        out_vts_path::AbstractString;
        nr::Int = 120,
        ntheta::Int = 72,
        nphi::Int = 144,
        r_inner::Real = Rinner,
        r_outer::Real = Router,
        polar_repair::Bool = true,
        compress::Bool = false,
    )
    nat = load_batsrus_tecplot(dat_path; polar_repair = polar_repair)
    g = _resample_batsrus_grid(nat; nr = nr, ntheta = ntheta, nphi = nphi, r_inner = r_inner, r_outer = r_outer)

    coords = Array{Float64, 4}(undef, 3, nr, ntheta, nphi)
    for k in 1:nphi, j in 1:ntheta, i in 1:nr
        sin_th, cos_th = sincos(g.theta[j])
        sin_ph, cos_ph = sincos(g.phi[k])
        ri = g.r[i]
        coords[1, i, j, k] = ri * sin_th * cos_ph
        coords[2, i, j, k] = ri * sin_th * sin_ph
        coords[3, i, j, k] = ri * cos_th
    end

    out_base = endswith(lowercase(out_vts_path), ".vts") ? out_vts_path[1:(end - 4)] : String(out_vts_path)
    mkpath(dirname(abspath(out_base)))

    saved = vtk_grid(out_base, coords; compress = compress, append = false) do vtk
        vtk["r_axis [m]", VTKFieldData()] = g.r
        vtk["theta_axis [rad]", VTKFieldData()] = g.theta
        vtk["phi_axis [rad]", VTKFieldData()] = g.phi
        vtk["Planet_radius [m]", VTKFieldData()] = [Rm]

        vtk["B_Field [T]", VTKPointData()] = g.B_T
        vtk["E_Total [V/m]", VTKPointData()] = g.E_total
        vtk["E_conv [V/m]", VTKPointData()] = g.E_conv
        vtk["E_hall [V/m]", VTKPointData()] = g.E_hall
        vtk["E_gradpe [V/m]", VTKPointData()] = g.E_gradpe
        vtk["E_res [V/m]", VTKPointData()] = g.E_res
        vtk["E_plot [V/m]", VTKPointData()] = g.E_plot

        vtk["n_H^p [m^-3]", VTKPointData()] = g.n_Hp
        vtk["n_Hsw^p [m^-3]", VTKPointData()] = g.n_Hswp
        vtk["n_Hp^p [m^-3]", VTKPointData()] = g.n_Hpp
        vtk["U_H^p [m/s]", VTKPointData()] = g.U_ms
        vtk["T_H^p [K]", VTKPointData()] = g.T_i

        vtk["n_O^p^p [m^-3]", VTKPointData()] = g.n_Op
        vtk["U_O^p^p [m/s]", VTKPointData()] = g.U_ms
        vtk["T_O^p^p [K]", VTKPointData()] = g.T_i

        vtk["n_O^2^p [m^-3]", VTKPointData()] = g.n_O2p
        vtk["U_O^2^p [m/s]", VTKPointData()] = g.U_ms
        vtk["T_O^2^p [K]", VTKPointData()] = g.T_i

        vtk["n_C^O^2^p [m^-3]", VTKPointData()] = g.n_CO2p
        vtk["U_C^O^2^p [m/s]", VTKPointData()] = g.U_ms
        vtk["T_C^O^2^p [K]", VTKPointData()] = g.T_i

        vtk["T_e [K]", VTKPointData()] = g.T_e
        vtk["n_e [m^-3]", VTKPointData()] = g.n_e
    end
    return abspath(only(saved))
end

"""
    interpolate_batsrus_slice(dat_path, u_grid, v_grid; plane="XZ", variable="O2p", slice_coord=0.0)

Interpolate a 2D slice (`"XZ"`, `"XY"`, or `"YZ"` in `Rm`) of a BATSRUS `.dat` file onto
`(u_grid, v_grid)` using `FastInterpolations` on the reconstructed 3D spherical grid.
"""
function interpolate_batsrus_slice(
        dat_path::AbstractString,
        u_grid::AbstractVector{<:Real},
        v_grid::AbstractVector{<:Real};
        plane::AbstractString = "XZ",
        variable::AbstractString = "O2p",
        slice_coord::Real = 0.0,
    )
    resolved = resolve_project_path(dat_path)
    hdr, data, _ = Batsrus.readtecdata(resolved)
    ix_x = _find_var_index(hdr.variable, ("x",))
    ix_y = _find_var_index(hdr.variable, ("y",))
    ix_z = _find_var_index(hdr.variable, ("z",))
    ix_var = _find_var_index(hdr.variable, (variable,))

    x_R = @view data[ix_x, :]
    y_R = @view data[ix_y, :]
    z_R = @view data[ix_z, :]
    vals = @view data[ix_var, :]

    pole_mask = (x_R .== 0.0f0) .& (y_R .== 0.0f0)
    r_R = sort(unique(Float64.(abs.(z_R[pole_mask]))))
    nr = length(r_R)
    ntheta = 61
    nphi_unique = 120
    nphi = nphi_unique + 1
    theta = collect(range(0.0, pi, length = ntheta))
    phi = collect(range(0.0, 2pi, length = nphi))
    dtheta = pi / (ntheta - 1)
    dphi = 2pi / nphi_unique

    arr = zeros(Float64, nr, ntheta, nphi)
    pole_counts = zeros(Int, nr, 2)
    for k in axes(data, 2)
        xk = Float64(x_R[k])
        yk = Float64(y_R[k])
        zk = Float64(z_R[k])
        rk = hypot(xk, yk, zk)
        ir = _nearest_sorted_index(r_R, rk)
        thk = acos(clamp(zk / rk, -1.0, 1.0))
        ith = clamp(round(Int, thk / dtheta) + 1, 1, ntheta)
        vk = Float64(vals[k])
        if ith == 1 || ith == ntheta
            pidx = ith == 1 ? 1 : 2
            c = pole_counts[ir, pidx] + 1
            pole_counts[ir, pidx] = c
            arr[ir, ith, 1] += (vk - arr[ir, ith, 1]) / c
        else
            phk = mod2pi(atan(yk, xk))
            iph = mod(round(Int, phk / dphi), nphi_unique) + 1
            arr[ir, ith, iph] = vk
        end
    end
    for ir in 1:nr, iph in 2:nphi
        arr[ir, 1, iph] = arr[ir, 1, 1]
        arr[ir, ntheta, iph] = arr[ir, ntheta, 1]
    end
    for ith in 2:(ntheta - 1), ir in 1:nr
        arr[ir, ith, nphi] = arr[ir, ith, 1]
    end

    itp = TP.build_interpolator(TP.StructuredGrid, arr, r_R, theta, phi)
    p = uppercase(String(plane))
    p in ("XZ", "XY", "YZ") || throw(ArgumentError("plane must be XZ, XY, or YZ"))
    w0 = Float64(slice_coord)
    nu, nv = length(u_grid), length(v_grid)
    out = Matrix{Float64}(undef, nu, nv)
    for j in 1:nv, i in 1:nu
        u = Float64(u_grid[i])
        v = Float64(v_grid[j])
        pt = p == "XZ" ? SA[u, w0, v] : p == "XY" ? SA[u, v, w0] : SA[w0, u, v]
        out[i, j] = itp(pt)
    end
    return out
end

function _select_batsrus_electric(g, electric_field)
    electric_field in (:total, "total", "E_Total [V/m]", "E_total [V/m]", "E [V/m]") && return copy(g.E_total)
    electric_field in (:conv, :convection, "conv", "convection", "E_conv [V/m]", "Econv") && return copy(g.E_conv)
    electric_field in (:hall, "hall", "E_hall [V/m]", "Ehall") && return copy(g.E_hall)
    electric_field in (:gradpe, "gradpe", "E_gradpe [V/m]", "Egradpe") && return copy(g.E_gradpe)
    electric_field in (:res, :resistive, "res", "resistive", "E_res [V/m]", "Eres") && return copy(g.E_res)
    electric_field in (:plot, "plot", "E_plot [V/m]", "Eplot") && return copy(g.E_plot)
    error("Unknown electric field selector: $electric_field")
end

function load_mhd_fields(
        path = data_path("mars_fields_spherical_from_dat.vts");
        electric_field = :total,
    )
    resolved = resolve_project_path(path)
    if endswith(lowercase(resolved), ".dat")
        nat = load_batsrus_tecplot(resolved)
        g = _resample_batsrus_grid(nat)
        B = copy(g.B_T)
        E = _select_batsrus_electric(g, electric_field)
        _rotate_vectors_to_spherical!(B, g.r, g.theta, g.phi)
        _rotate_vectors_to_spherical!(E, g.r, g.theta, g.phi)
        return MHDFields(g.r, g.theta, g.phi, E, B, Symbol(electric_field), resolved)
    end

    vtk = VTKFile(resolved)
    point_data = get_point_data(vtk)
    dims = ReadVTK.get_wholeextent(vtk.xml_file)[1]
    nr, ntheta, nphi = dims
    r, theta, phi = _mhd_axes(vtk, nr, ntheta, nphi)

    B_raw = if "B_Field [T]" in keys(point_data)
        get_data(point_data["B_Field [T]"])
    elseif "B [nT]" in keys(point_data)
        get_data(point_data["B [nT]"]) .* 1.0e-9
    elseif "B" in keys(point_data)
        get_data(point_data["B"]) .* 1.0e-9
    else
        error("Magnetic field not found in $resolved")
    end
    E_raw = _point_vector(point_data, _electric_names(electric_field))

    B = Float64.(reshape(B_raw, 3, nr, ntheta, nphi))
    E = Float64.(reshape(E_raw, 3, nr, ntheta, nphi))
    _rotate_vectors_to_spherical!(B, r, theta, phi)
    _rotate_vectors_to_spherical!(E, r, theta, phi)

    return MHDFields(r, theta, phi, E, B, Symbol(electric_field), resolved)
end

function _read_array(point_data, name, nr, ntheta, nphi)
    values = get_data(point_data[name])
    if length(values) == nr * ntheta * nphi
        return Float64.(reshape(values, nr, ntheta, nphi))
    elseif length(values) == 3 * nr * ntheta * nphi
        return Float64.(reshape(values, 3, nr, ntheta, nphi))
    end
    error("Unexpected array size for $name")
end

function load_mhd_moments(path = data_path("mars_fields_spherical_from_dat.vts"))
    resolved = resolve_project_path(path)
    if endswith(lowercase(resolved), ".dat")
        nat = load_batsrus_tecplot(resolved)
        g = _resample_batsrus_grid(nat)
        return (;
            r = g.r,
            theta = g.theta,
            phi = g.phi,
            Hplus = SpeciesMoments(g.n_Hp, copy(g.U_ms), copy(g.T_i)),
            Oplus = SpeciesMoments(g.n_Op, copy(g.U_ms), copy(g.T_i)),
            O2plus = SpeciesMoments(g.n_O2p, copy(g.U_ms), copy(g.T_i)),
            CO2plus = SpeciesMoments(g.n_CO2p, copy(g.U_ms), copy(g.T_i)),
            Te = g.T_e,
        )
    end

    vtk = VTKFile(resolved)
    point_data = get_point_data(vtk)
    nr, ntheta, nphi = ReadVTK.get_wholeextent(vtk.xml_file)[1]
    r, theta, phi = _mhd_axes(vtk, nr, ntheta, nphi)

    moments = (
        Hplus = SpeciesMoments(
            _read_array(point_data, "n_H^p [m^-3]", nr, ntheta, nphi),
            _read_array(point_data, "U_H^p [m/s]", nr, ntheta, nphi),
            _read_array(point_data, "T_H^p [K]", nr, ntheta, nphi),
        ),
        Oplus = SpeciesMoments(
            _read_array(point_data, "n_O^p^p [m^-3]", nr, ntheta, nphi),
            _read_array(point_data, "U_O^p^p [m/s]", nr, ntheta, nphi),
            _read_array(point_data, "T_O^p^p [K]", nr, ntheta, nphi),
        ),
        O2plus = SpeciesMoments(
            _read_array(point_data, "n_O^2^p [m^-3]", nr, ntheta, nphi),
            _read_array(point_data, "U_O^2^p [m/s]", nr, ntheta, nphi),
            _read_array(point_data, "T_O^2^p [K]", nr, ntheta, nphi),
        ),
        CO2plus = SpeciesMoments(
            _read_array(point_data, "n_C^O^2^p [m^-3]", nr, ntheta, nphi),
            _read_array(point_data, "U_C^O^2^p [m/s]", nr, ntheta, nphi),
            _read_array(point_data, "T_C^O^2^p [K]", nr, ntheta, nphi),
        ),
        Te = _read_array(point_data, "T_e [K]", nr, ntheta, nphi),
    )
    return (; r, theta, phi, moments...)
end

function mhd_param(fields::MHDFields; species = "O2+", erased::Bool = false)
    sp = species isa TP.Species ? species : TP.SpeciesDict[String(species)]
    E_itp = TP.build_interpolator(TP.StructuredGrid, fields.E, fields.r, fields.theta, fields.phi)
    B_itp = TP.build_interpolator(TP.StructuredGrid, fields.B, fields.r, fields.theta, fields.phi)
    return TP.prepare(; E = E_itp, B = B_itp, species = sp, erased = erased)
end

"""MHD O2+ scalar interpolators; velocity components remain Cartesian (m/s)."""
struct IonosphereSource{N, T, U}
    n::N
    temperature::T
    velocity::U
    radius::Float64
end

function _ionosphere_radius(altitude_km)
    isfinite(altitude_km) && 200 <= altitude_km <= 800 ||
        throw(ArgumentError("ionosphere_altitude_km must be in [200, 800]"))
    return Rm + 1e3 * altitude_km
end

function load_ionosphere_source(
        path = data_path("mars_fields_spherical_from_dat.vts");
        altitude_km = 400.0,
    )
    radius = _ionosphere_radius(altitude_km)
    moments = load_mhd_moments(path)
    r, theta, phi = moments.r, moments.theta, moments.phi
    scalar(A) = TP.build_interpolator(TP.StructuredGrid, A, r, theta, phi)
    n = moments.O2plus.n
    T = moments.O2plus.t
    U = moments.O2plus.v
    # Scalar interpolation prevents the vector interpolator from interpreting
    # these Cartesian components as spherical vector components.
    return IonosphereSource(
        scalar(n),
        scalar(T),
        ntuple(k -> scalar(Array(U[k, :, :, :])), 3),
        radius,
    )
end

"""Sample n [m^-3], Ti [K], Ui [m/s], and n*norm(Ui) [m^-2 s^-1]."""
function ionosphere_properties(source::IonosphereSource, position)
    all(isfinite, position) && norm(position) > 0 ||
        throw(ArgumentError("Expected a finite nonzero position"))
    # Project onto the configured shell, with a nanometre-scale inward-grid
    # offset only at the 200 km endpoint to avoid roundoff outside the table.
    x = position / norm(position) * max(source.radius, Rinner + 1e-8)
    n, Ti = source.n(x), source.temperature(x)
    Ui = SVector{3, Float64}(ntuple(k -> source.velocity[k](x), 3))
    isfinite(n) && n >= 0 && isfinite(Ti) && Ti > 0 && all(isfinite, Ui) ||
        error("Invalid MHD O2+ moments at ionosphere position $x: n=$n, Ti=$Ti, Ui=$Ui")
    return (; n, Ti, Ui, flux = n * norm(Ui))
end

function ionosphere_distribution(
        source::IonosphereSource,
        position,
        velocity;
        mass = TP.SpeciesDict["O2+"].m,
    )
    moments = ionosphere_properties(source, position)
    vth = sqrt(2 * TP.kB * moments.Ti / mass)
    g = VDF.Maxwellian(vth; u0 = moments.Ui)(velocity)
    return (; f = moments.n * g, g, moments...)
end
