# Mars MHD fields for test-particle calculations

This directory contains two BATSRUS Mars simulations prepared for test-particle calculations. Both use the same grid, upstream solar-wind conditions, chemistry, resistivity, and separate ion/electron pressures. The simulations are steady-state iterations (`DoTimeAccurate = F`), so the numbers in the filenames (n=120k) are iteration numbers rather than physical times.

## Cases

- `MSPe`: multispecies MHD with a separate electron-pressure equation. Hall resistivity is off. The electron-pressure-gradient contribution is also excluded.
- `MSPeHall`: the same setup with Hall resistivity on (`HallFactor = 1.0`, `HallCmaxFactor = 0.5`). In this BATSRUS implementation, using Hall resistivity also includes the electron-pressure-gradient contribution in the induction equation.
- `MHD-with-crustal`: MHD steady state solution with a crustal field.
- `MHD-without-crustal`: MHD steady state solution without a crustal field.

## Variables and units

The variable list is stored in the Tecplot header. It contains:

- Coordinates: `X`, `Y`, `Z` [Mars radii]
- Total mass densities: `Rho` [amu/cm^3]
- Ion species number density: `Hswp`, `Hpp`, `O2p`, `Op`, `Co2p` [cm^-3]
- Bulk ion velocity: `U_x`, `U_y`, `U_z` [km/s]
- Magnetic field: `B_x`, `B_y`, `B_z` [nT]
- Hyperbolic divergence-cleaning field, only for numerical purpose: `Hyp` [nT]
- Electron and ion pressures: `pe`, `P` [nPa]
- Current density: `J_x`, `J_y`, `J_z` [mA/m^2]
- Magnetic diffusivity: `eta` [m^2/s]
- Electron-pressure gradient: `GradPe_x`, `GradPe_y`, `GradPe_z` [nPa/m]
- Electric-field diagnostic: `E_x`, `E_y`, `E_z` [mV/m]

`Hswp` is the solar-wind proton population and `Hpp` is the planetary proton population.

## Important electric-field warning

The saved `E_x`, `E_y`, and `E_z` are plotting diagnostics, not the exact electric field used by the induction equation. BATSRUS calculates these output variables as

```text
E_plot = -u_e x B,
u_e = u_ion - J/(n_e e).
```

The plotting routine forces the current correction on in both datasets, even in the non-Hall simulation. It also does not include the complete pressure-gradient, resistive, and numerical-flux contributions.

For quantitative comparison, reconstruct the electric field consistently from other variables, for example

```text
E = -u_ion x B + (J x B)/(n_e e) - grad(Pe)/(n_e e) + eta*mu0*J,
```

with terms included or excluded according to the selected MHD case. The species densities can be used to calculate the electron charge density under quasineutrality. Please do not treat `E_x/E_y/E_z` as the definitive Hall-run electric field without this reconstruction/check.
