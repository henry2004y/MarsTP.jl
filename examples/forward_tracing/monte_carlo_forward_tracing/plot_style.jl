# Shared publication styling helper for all Mars O2+ manuscript figures (Fig. 1–8 & Fig. A1).
#
# Standardizes:
#   - Font family: Arial sans-serif (fallback DejaVu Sans) + default DejaVu Sans mathtext
#   - Width-scaled typography targeting uniform printed font sizes at \textwidth (~6.7 in)
#   - Colormaps:
#       * Continuous:     turbo (masked/unoccupied bins in dark gray "0.15")
#       * Symmetric at 0: RdBu_r (masked/unoccupied bins in dark gray "0.15")
#       * Discrete:       ColorBrewer Set2 (Convective = Set2[3] blue, Hall = Set2[2] orange,
#                         Ambipolar = Set2[1] teal; Total / MHD baseline = black)
#   - Reference lines: black or gray (0.45 on light/diverging backgrounds; 0.78 on dark/turbo maps)

using PyPlot

const PAPER_TEXTWIDTH_IN = 6.7
const PAPER_BAD_COLOR    = "0.15"

# ColorBrewer Set2 palette (8 qualitative colors)
const SET2_TEAL   = "#66c2a5" # Set2[1]: Ambipolar (E_A, ε_A, P_A)
const SET2_ORANGE = "#fc8d62" # Set2[2]: Hall (E_H, ε_H, P_H)
const SET2_BLUE   = "#8da0cb" # Set2[3]: Convective (E_C, ε_C, P_C)
const SET2_PINK   = "#e78ac3" # Set2[4]: 4th discrete series / sampling accent
const SET2_LIME   = "#a6d854" # Set2[5]
const SET2_YELLOW = "#ffd92f" # Set2[6]
const SET2_TAN    = "#e5c494" # Set2[7]
const SET2_GRAY   = "#b3b3b3" # Set2[8]

const SET2_PALETTE = (
    SET2_TEAL,
    SET2_ORANGE,
    SET2_BLUE,
    SET2_PINK,
    SET2_LIME,
    SET2_YELLOW,
    SET2_TAN,
    SET2_GRAY,
)

# Canonical electric-field channel colors
const COL_CONV   = SET2_BLUE
const COL_HALL   = SET2_ORANGE
const COL_AMB    = SET2_TEAL
const COL_TOTAL  = "black"
const COL_ACCENT = SET2_PINK   # Virtual orbit / transect / warped CS highlight (distinct from black/gray ref lines)

# Canonical reference-line colors (strictly black or gray)
const REF_BLACK   = "black"    # Primary structural boundaries on white (BS, MPB)
const REF_GRAY    = "0.45"     # Zero lines, guides, and dividers on white or RdBu_r
const REF_ON_DARK = "0.78"     # Reference lines over dark/turbo 2D maps

"""
    paper_style(fig_width_in::Real; target_width_in::Real = PAPER_TEXTWIDTH_IN)

Configure `PyPlot.matplotlib.rcParams` for publication-quality rendering and return a
NamedTuple of width-scaled font sizes, tick sizes, and line widths so all figures share
identical apparent typography when scaled to `\\textwidth` in the manuscript.
"""
function paper_style(fig_width_in::Real; target_width_in::Real = PAPER_TEXTWIDTH_IN, font_scale::Real = 1.0)
    s = (Float64(fig_width_in) / Float64(target_width_in)) * Float64(font_scale)

    fs_panel    = round(9.2 * s; digits = 1)
    fs_title    = round(8.8 * s; digits = 1)
    fs_title_sm = round(8.0 * s; digits = 1)
    fs_label    = round(8.6 * s; digits = 1)
    fs_cbar     = round(8.2 * s; digits = 1)
    fs_tick     = round(7.8 * s; digits = 1)
    fs_legend   = round(7.2 * s; digits = 1)
    fs_annot    = round(6.8 * s; digits = 1)
    fs_annot_sm = round(6.0 * s; digits = 1)

    maj_sz = round(3.0 * s; digits = 1)
    min_sz = round(1.8 * s; digits = 1)
    maj_w  = round(0.60 * s; digits = 2)
    min_w  = round(0.46 * s; digits = 2)

    rc = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rc["font.family"]       = "sans-serif"
    rc["font.sans-serif"]   = ["Arial", "DejaVu Sans"]
    rc["mathtext.fontset"]  = "dejavusans"
    rc["font.size"]         = fs_tick
    rc["axes.labelsize"]    = fs_label
    rc["axes.titlesize"]    = fs_title
    rc["xtick.labelsize"]   = fs_tick
    rc["ytick.labelsize"]   = fs_tick
    rc["xtick.major.size"]  = maj_sz
    rc["ytick.major.size"]  = maj_sz
    rc["xtick.major.width"] = maj_w
    rc["ytick.major.width"] = maj_w
    rc["xtick.minor.size"]  = min_sz
    rc["ytick.minor.size"]  = min_sz
    rc["xtick.minor.width"] = min_w
    rc["ytick.minor.width"] = min_w
    rc["legend.fontsize"]   = fs_legend

    return (
        scale       = s,
        panel       = fs_panel,
        title       = fs_title,
        title_sm    = fs_title_sm,
        label       = fs_label,
        cbar        = fs_cbar,
        tick        = fs_tick,
        legend      = fs_legend,
        annot       = fs_annot,
        annot_sm    = fs_annot_sm,
        maj_sz      = maj_sz,
        min_sz      = min_sz,
        maj_w       = maj_w,
        min_w       = min_w,
        lw_curve    = 2.4,
        lw_total    = 2.5,
        lw_ref      = 1.1,
    )
end

"""
    paper_cmap_continuous(; bad = PAPER_BAD_COLOR)

Return a copy of the canonical continuous colormap (`turbo`) with `bad` masked to `PAPER_BAD_COLOR`.
"""
function paper_cmap_continuous(; bad::AbstractString = PAPER_BAD_COLOR)
    cm = matplotlib.cm.get_cmap("turbo").copy()
    cm.set_bad(color = bad)
    return cm
end

"""
    paper_cmap_diverging(; bad = PAPER_BAD_COLOR)

Return a copy of the canonical symmetric-at-zero colormap (`RdBu_r`) with `bad` masked to `PAPER_BAD_COLOR`.
"""
function paper_cmap_diverging(; bad::AbstractString = PAPER_BAD_COLOR)
    cm = matplotlib.cm.get_cmap("RdBu_r").copy()
    cm.set_bad(color = bad)
    return cm
end

"""
    style_colorbar!(cb, st; label = "")

Apply consistent tick and label styling to a PyPlot colorbar `cb` using style `st = paper_style(...)`.
"""
function style_colorbar!(cb, st; label::AbstractString = "", labelpad::Real = 6)
    cb.ax.tick_params(which = "major", labelsize = st.tick, length = st.maj_sz * 0.85, width = st.maj_w)
    cb.ax.tick_params(which = "minor", length = st.min_sz * 0.85, width = st.min_w)
    if !isempty(label)
        cb.set_label(label; fontsize = st.cbar, labelpad = labelpad)
    end
    return cb
end
