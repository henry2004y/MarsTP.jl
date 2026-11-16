# Source Altitude Sensitivity and Test-Particle vs. MHD Discrepancy

![Unified 3-panel 1D TestParticle.jl vs. MHD benchmark](images/tp_mhd_discrepancy_6panel.png)

Because boundary-injected $\mathrm{O}_2^+$ test particles ($m = 32\ \mathrm{amu}$) use Liouville-consistent one-way outward surface-flux weights ($w_i \propto v_n f_M(\mathbf{V})$, Manuscript Equations 2–4), the normal speed $v_n$ cancels the residence time $\tau_i = \Delta h / v_n$ in the virtual detector (Manuscript Equation 6), eliminating boundary-sampling bias and bringing 3D global shell densities within $10\%$–$14\%$ of 3D MHD across $0.50$–$1.00\,R_M$. Even when electromagnetic fields are in exact steady state and boundary sampling is exact, reconstructed collisionless test-particle densities and velocities still depart physically from global MHD fluid profiles and depend on the launch altitude $h_0$ due to two remaining mechanisms, while relative electric-field energization shares ($\langle \varepsilon_C \rangle, \langle \varepsilon_H \rangle, \langle \varepsilon_A \rangle$) remain comparable across launch altitudes.

This example uses a unified 1D analytical model of the Martian dayside ionosphere ($h \in [150, 920]\ \mathrm{km}$, plotted to $900\ \mathrm{km}$, $g = 0$, matching Manuscript Equations 1–6) implemented in [`src/analysis/tp_mhd_discrepancy.jl`](../../src/analysis/tp_mhd_discrepancy.jl) and run via [`tp_mhd_discrepancy_benchmark.jl`](tp_mhd_discrepancy_benchmark.jl) with exact surface-flux boundary sampling (`sample_mode = :flux`).

---

## 1. Physical Model: 1D Ionospheric Outflow Columns

In steady state, the 1D MHD continuity and vertical momentum equations for scalar-pressure $\mathrm{O}_2^+$ outflow (with constant-$T_i$ benchmark limit $T_i = 1500\ \mathrm{K}$, $T_e = 2000\ \mathrm{K}$, $v_{\mathrm{th}} = \sqrt{k_B T_i / m} \approx 0.62\ \mathrm{km/s}$) are:

$$
\frac{d}{dh}\left(n_{\mathrm{MHD}} U_h\right) = S_{\mathrm{net}}(h) = P(h) - L(h), \qquad
m n_{\mathrm{MHD}} U_h \frac{dU_h}{dh} = \left(\mathbf{J}\times\mathbf{B}\right)_h - \frac{dP_e}{dh} - \frac{dP_i}{dh}
$$

Using the generalized Ohm's law with massless electrons ($m_e \to 0$):

$$
\mathbf{E} = -\mathbf{U}_i \times \mathbf{B} + \frac{\mathbf{J}\times\mathbf{B}}{q n_e} - \frac{\nabla P_e}{q n_e} = \mathbf{E}_C + \mathbf{E}_H + \mathbf{E}_A
$$

Substituting $\mathbf{E}$ into the fluid momentum equation yields:

$$
m U_h \frac{dU_h}{dh} = q\left(\mathbf{E} + \mathbf{U}\times\mathbf{B}\right)_h - \frac{1}{n_{\mathrm{MHD}}}\frac{dP_i}{dh}
$$

---

## 2. Two Fundamental Physical Sources of TP–MHD Discrepancy

### Mechanism 1: Fluid Scalar-Pressure Closure vs. Kinetic Lorentz Streaming (Panel a)

Above the lower-ionospheric collision layer ($\lesssim 200\ \mathrm{km}$), both global space-plasma MHD and test-particle models assume a **collisionless** plasma, but they close the Vlasov moment hierarchy differently. In collisionless kinetic models (both test-particle tracing and hybrid-PIC simulations), individual ions obey deterministic scatter-free Lorentz motion $m\,d\mathbf{V}_i/dt = q(\mathbf{E} + \mathbf{V}_i\times\mathbf{B})$ and **never** receive an artificial single-particle $-\nabla P_i/(qn)$ push. In kinetic theory, $-\nabla\cdot\mathbf{P}_i$ enters the momentum equation solely as the velocity moment of spatial free-streaming ($\mathbf{v}\cdot\nabla f_i$). While massless electrons ($m_e \to 0$) turn $-\nabla P_e$ into a true electrostatic field $\mathbf{E}_A = -\nabla P_e / (q n_e)$ in Ohm's law that accelerates every test ion via $q\mathbf{E}_A$, deterministic velocity bunching during scatter-free expansion narrows the parallel ion velocity spread ($T_\parallel$ decreases anisotropically). By contrast, five-moment MHD truncates the hierarchy with an isotropic scalar-pressure closure (adiabatic $P_i \propto n^\gamma$ with $\gamma = 5/3$ in 3D, or the constant-$T_i$ limit $P_i = n k_B T_i$ in this 1D benchmark—implicitly approximating Larmor gyrotropy and wave–particle pitch-angle isotropization with zero heat flux) so $-\nabla P_i$ accelerates fluid elements directly (and in multi-species 3D MHD, $\mathbf{E}_C = -\mathbf{U}_i \times \mathbf{B}$ also uses the center-of-mass ion velocity $\mathbf{U}_i \neq \mathbf{U}_{\mathrm{O}_2^+}$).

Panel **(a)** demonstrates this closure difference using `benchmark_parallel_gradpi` for steady expansion along a parallel vertical field $\mathbf{B} = B_h\hat{\mathbf{h}}$ ($B_h = 15\ \mathrm{nT}$, $H_n = 140\ \mathrm{km}$, $U_0 = 2.0\ \mathrm{km/s}$ at $h_0 = 200\ \mathrm{km}$):
- With $E_\parallel = 0$, free-streaming ions coast at $U_h = 2.00\ \mathrm{km/s}$.
- Kinetic test particles accelerated solely by $\mathbf{E}_A = -\nabla P_e / (q n)$ reach $3.01\ \mathrm{km/s}$ at $800\ \mathrm{km}$.
- The MHD scalar-pressure closure reaches $U_{h,\mathrm{MHD}}(800\ \mathrm{km}) = 3.42\ \mathrm{km/s}$ because of the imposed scalar $-\nabla P_i$ fluid closure (indicated by the double-headed arrow).

### Mechanism 2: Ionospheric $\mathrm{O}_2^+$ Source Layer, Monotonic Outflow Structure, and Why $h_0 = 400\ \mathrm{km}$ Is Optimal (Panels b–c)

In local horizontal–vertical unit vectors $(\hat{\mathbf{e}}_1, \hat{\mathbf{e}}_2, \hat{\mathbf{h}})$, the oblique draped magnetic pileup column is $\mathbf{B}(h) = B_1(h)\hat{\mathbf{e}}_1 + B_h\hat{\mathbf{h}}$ ($B_h = 1.2\ \mathrm{nT}$, $B_{1,\mathrm{top}} = 15\ \mathrm{nT}$ at $920\ \mathrm{km}$, $B_1$ decreasing upward), with $\mathbf{J} = \frac{1}{\mu_0}\frac{dB_1}{dh}\hat{\mathbf{e}}_2$ driving a self-consistent Hall electric field $\mathbf{E}_H = (\mathbf{J}\times\mathbf{B})/(q n_e)$ alongside $\mathbf{E}_C$ and $\mathbf{E}_A$. Rather than imposing a velocity curve independently and producing an unphysical sub-$200\ \mathrm{km}$ density kink, we prescribe both $S_{\mathrm{net}}(h)$ (calibrated to steady-state 3D BATS-R-US dayside outflow scales) and a smooth, strictly monotonically decreasing ionospheric + MPB-transition + sheath density profile referenced to $h_{\mathrm{ref}} = 200\ \mathrm{km}$:

$$
n_{\mathrm{MHD}}(h) = n_{\mathrm{iono}} e^{-(h - h_{\mathrm{ref}})/H_{\mathrm{iono}}} + n_{\mathrm{mpb}} e^{-(h - h_{\mathrm{ref}})/H_{\mathrm{mpb}}} + n_{\mathrm{asymp}}, \qquad
U_h(h) = \frac{\Gamma_{\mathrm{MHD}}(h)}{n_{\mathrm{MHD}}(h)}
$$

which guarantees $\frac{dn_{\mathrm{MHD}}}{dh} < 0$ and $\frac{dU_h}{dh} = \frac{S_{\mathrm{net}} - U_h\,(dn_{\mathrm{MHD}}/dh)}{n_{\mathrm{MHD}}} > 0$ everywhere on $[150, 920]\ \mathrm{km}$. The analytical parameters ($S_0 = 6.2 \times 10^6\ \mathrm{m^{-3}\,s^{-1}}$, $h_{\mathrm{base}} = 185\ \mathrm{km}$, $h_{\mathrm{peak}} = 205\ \mathrm{km}$, $H_{\mathrm{rise}} = 20\ \mathrm{km}$, $H_{\mathrm{decay}} = 40\ \mathrm{km}$, $\Gamma_{\mathrm{base}} = 0.25 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$, $n_{\mathrm{iono}} = 1.45 \times 10^8\ \mathrm{m^{-3}}$, $H_{\mathrm{iono}} = 46\ \mathrm{km}$, $n_{\mathrm{mpb}} = 1.15 \times 10^8\ \mathrm{m^{-3}}$, $H_{\mathrm{mpb}} = 125\ \mathrm{km}$, $n_{\mathrm{asymp}} = 0.26 \times 10^8\ \mathrm{m^{-3}}$, $\alpha_{\mathrm{Hall}} = 0.32$) yield peak $S_{\mathrm{net}} = 3.93 \times 10^6\ \mathrm{m^{-3}\,s^{-1}}$ near $207\ \mathrm{km}$ ($90.27\%$ of $>200\ \mathrm{km}$ production below $300\ \mathrm{km}$ and $99.20\%$ below $400\ \mathrm{km}$; orange shading in Panel **b**), $n_{\mathrm{MHD}} = (2.86, 0.51, 0.31, 0.27) \times 10^8\ \mathrm{m^{-3}}$ and $U_h = (0.21, 5.79, 9.69, 11.04)\ \mathrm{km/s}$ at $200, 400, 600, 800\ \mathrm{km}$, with $\Gamma_{\mathrm{MHD}}(800\ \mathrm{km}) = 2.98 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$. (In the 3D MHD background itself, $\nabla\cdot(n_{\mathrm{O}_2^+}\mathbf{U}_i)$ similarly turns positive above $\sim 185\ \mathrm{km}$ and peaks at $2.65 \times 10^6\ \mathrm{m^{-3}\,s^{-1}}$ near $203$–$209\ \mathrm{km}$ with a secondary shoulder near $285\ \mathrm{km}$, placing $70\%$–$78\%$ of $>200\ \mathrm{km}$ net production below $300\ \mathrm{km}$ and $97.4\%$–$100\%$ below $400\ \mathrm{km}$, while $\sim 96\%$ of low-altitude production convects downward to recombine so global net radial outflow emerges above $\sim 350\ \mathrm{km}$.)

Bracketing the $400\ \mathrm{km}$ launch shell on both sides at $h_c = 800\ \mathrm{km}$ ($N_p = 15{,}000$, $154$ bins) demonstrates why **$h_0 = 400\ \mathrm{km}$** is the optimal initialization altitude rather than $200\ \mathrm{km}$ or $600\ \mathrm{km}$:

- **$h_0 = 200\ \mathrm{km}$ (too low):** Sits inside the photochemical production layer ($U_h = 0.21\ \mathrm{km/s} < v_{\mathrm{th}}$) adjacent to the $200\ \mathrm{km}$ inner absorbing boundary; it misses the volumetric $\mathrm{O}_2^+$ source over $200$–$400\ \mathrm{km}$ ($\Gamma_{\mathrm{MHD}}(200\ \mathrm{km}) = 0.61 \times 10^{11}$ vs. $\Gamma_{\mathrm{MHD}}(400\ \mathrm{km}) = 2.96 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$) and loses $18.4\%$ of trajectories to first-gyration boundary absorption, recovering only $\Gamma_{800} = 0.86 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$ ($28.9\%$ of MHD).
- **$h_0 = 600\ \mathrm{km}$ (too high):** Gains only $0.8\%$ extra source flux over $400\ \mathrm{km}$ ($\Gamma_{800} = 2.98 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$), yet leaves the entire $400$–$600\ \mathrm{km}$ upper-ionosphere/MPB transition unmodeled, injects an artificial isotropic Maxwellian where real ions are already supersonic ($U_h = 9.69\ \mathrm{km/s} \approx 16\,v_{\mathrm{th}}$) and non-Maxwellian, and omits large low-altitude electric-field acceleration—cutting cumulative work $\langle\varepsilon_T\rangle_{800}$ by $>50\%$ ($8.79\ \mathrm{eV}$ vs. $18.35\ \mathrm{eV}$ at $400\ \mathrm{km}$ and $19.99\ \mathrm{eV}$ at $200\ \mathrm{km}$) and artificially skewing shares toward convective ($82.7\%$) at the expense of Hall ($17.0\%$) and ambipolar ($0.3\%$) work.
- **$h_0 = 400\ \mathrm{km}$ (optimal sweet spot):** Sits just above the photochemical source ($99.20\%$ complete) and boundary trapping zone (`100%` escape), recovering $\Gamma_{800} = 2.95 \times 10^{11}\ \mathrm{m^{-2}\,s^{-1}}$ ($99.0\%$ of MHD) while starting below the main pickup acceleration region so it preserves $91.8\%$ of total cumulative work ($\langle\varepsilon_T\rangle = 18.35\ \mathrm{eV}$) and the full low-altitude $\mathbf{E}_C / \mathbf{E}_H / \mathbf{E}_A$ partition.

---

## 3. Cumulative Electric-Field Work and Sensitivity at $h_c = 800\ \mathrm{km}$ (Panel c)

Comparing cumulative electric-field work ($\langle\varepsilon_C\rangle, \langle\varepsilon_H\rangle, \langle\varepsilon_A\rangle$) and total work $\langle\varepsilon_T\rangle$ at $h_c = 800\ \mathrm{km}$ across $h_0$:
- $h_0 = 200\ \mathrm{km}$ ($\langle\varepsilon_T\rangle = 19.99\ \mathrm{eV}$): $66.8\%$ convective, $31.2\%$ Hall, $2.0\%$ ambipolar
- $h_0 = 400\ \mathrm{km}$ ($\langle\varepsilon_T\rangle = 18.35\ \mathrm{eV}$): $74.0\%$ convective, $25.4\%$ Hall, $0.6\%$ ambipolar
- $h_0 = 600\ \mathrm{km}$ ($\langle\varepsilon_T\rangle = 8.79\ \mathrm{eV}$): $82.7\%$ convective, $17.0\%$ Hall, $0.3\%$ ambipolar

---

## 4. Running the Benchmark

From the repository root:

```sh
julia --threads=auto --project=. examples/forward_tracing/tp_mhd_discrepancy_benchmark.jl
```

This invokes `AnalyticalOutflowColumn`, `benchmark_weighting_vdf`, `benchmark_parallel_gradpi`, and `trace_outflow_column` from `MarsTP` and writes `examples/forward_tracing/images/tp_mhd_discrepancy_6panel.png`.

To regenerate Appendix Figure A1 directly in the companion `MarsO2+` manuscript repository:

```powershell
$env:TP_MHD_BENCHMARK_OUTPUT = "C:/Users/henry/Documents/Research/MarsO2+/figures/figA1.png"
julia --threads=auto --project=. examples/forward_tracing/tp_mhd_discrepancy_benchmark.jl
```
