# Data Files

This directory should contain only prepared runtime inputs.

| File | Description |
|---|---|
| `amps_sph.mat` | AMPS hot atomic oxygen spherical grid. |
| `gitm_sph.mat` | GITM neutral atmosphere spherical grid. |
| `O2plus_source_rates.mat` | Prepared O2+ production and outflow rates. |

The tracing code expects the converted MHD field files in `data/`:

```text
data/mars_fields_spherical_from_dat.vts
data/mars_fields_with_crustal.vts
data/mars_fields_without_crustal.vts
```

These `.vts` files (and the generated `data/O2plus_source_rates_with_crustal.mat` and `data/O2plus_source_rates_without_crustal.mat`) are not uploaded to GitHub because of GitHub's file-size limit. Generate them directly from the BATSRUS `.dat` files in `BATSRUS-data/` by running:

```sh
julia --project=. scripts/convert_batsrus_mhd.jl
```

Important fields in `O2plus_source_rates.mat`:

| Variable | Unit |
|---|---|
| `r_m`, `theta_rad`, `phi_rad` | m, rad, rad |
| `O2plus_production_density_m3_inv_s_inv` | m^-3 s^-1 |
| `O2plus_production_rate_s_inv` | s^-1 |
| `O2plus_outflow_rate_s_inv` | s^-1 |
| `cell_volume_m3` | m^3 |
| `surface_area_m2` | m^2 |
