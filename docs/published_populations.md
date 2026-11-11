# Published population prescriptions and radial-law alignment

The ad hoc 1–400 Msun experiment is **not** a published IMF prescription. It is retained as an archived sensitivity benchmark, not relabeled as Pan/Xue. The published-default switch was rolled back at the user's request. CLI, comparison workflow and programmatic defaults all use `legacy_popiii`; the published presets below are opt-in experimental alternatives only. Supplying custom Pop III IMF bounds/slope or SEVN without an explicit population choice selects the legacy path. Run metadata always records the resolved population.

## Mass functions and number normalization

| Preset | Initial objects | Normalization | Independent M_NSC? |
|---|---|---|---|
| `legacy_popiii` | Previous analytic stellar IMF and lifetime clocks | Sample to total M_NSC | Yes |
| `pan2021_nsc` | 1-Msun stars and pre-existing 10-Msun BHs, N_BH/N_star=1e-3 | M_NSC is total star+BH mass | Yes, conditional variant |
| `pan2021_paper` | Same Pan components | M_star=20 M_bullet; N_BH=1e-3 N_star | No |
| `xue2026_bh` | Pre-existing BHs, dN/dm proportional to m^-2.35 on 5–15 Msun | N_BH=floor(M_bullet/(20 Msun)) | No |

Pan uses two representative components, not a continuous 1–400 Msun stellar IMF. Xue's power law describes **BHs**, not all stellar mass in the NSC. Normalizing that BH law to all of M_NSC would substantially overcount BHs.

For `pan2021_nsc`, before integer rounding:

$$
N_\star=\frac{M_{\rm NSC}}{M_\odot+\delta\,10M_\odot},\qquad
N_{\rm BH}=\delta N_\star,\qquad \delta=10^{-3},
$$

$$
\langle m\rangle=\frac{1+10\delta}{1+\delta}M_\odot
\simeq1.009M_\odot.
$$

At M_bullet=1e5 and M_NSC=3e6 Msun, this gives **2,970,300 stars + 2,970 BHs = 2,973,270 objects**, exactly 3e6 Msun with the implemented integer allocation. The Pan paper's canonical normalization instead gives 2,000,000 stars + 2,000 BHs, with **total** component mass 2.02e6 Msun. Do not confuse its M_star=20 M_bullet with a total cluster mass of 20 M_bullet.

Xue's reference population contains 5,000 pre-existing BHs at M_bullet=1e5. Its sampled mean BH mass is about 7.9 Msun and component mass about 4e4 Msun; this is not a whole 3e6-Msun stellar cluster. The driver rejects an independent NSC-mass panel for `pan2021_paper` or `xue2026_bh` rather than drawing a fictitious dependency.

BH-formation clocks now reflect these populations: existing BHs have clock zero. Representative Pan stars have clock infinity (they do not automatically become BHs). These presets require the fixed-component backend and cannot be sent to a SEVN ZAMS-evolution calculation as if old BHs were newly born stars.

## Radial slopes: the measure matters

A spherical spatial number density obeys

$$
\frac{dN}{dr}=4\pi r^2n(r),\qquad
\frac{dN}{d\ln r}=r\frac{dN}{dr}.
$$

Consequently the following statements are different:

| Model | Spatial density | Radial number law | Log-radius number law |
|---|---|---|---|
| Legacy inner-cusp proxy, gamma=1.75 | Associated n(r) ~ r^-1.75 | p(a) ~ a^0.25 in the actual sampler | dN/dln(a) ~ a^1.25 |
| Pan inner asymptote, gamma=1.5 | n(r) ~ r^-1.5 | dN/dr ~ r^0.5 | dN/dln(r) ~ r^1.5 |
| Pan outer asymptote | n(r) ~ r^-4 | dN/dr ~ r^-2 | dN/dln(r) ~ r^-1 |
| Xue gamma_rho=0 | Spherical average n(r) ~ r^-2 | dN/dr ~ r^0 | dN/dln(r) ~ r |

Pan also reports evolved mass-segregated profiles: approximately r^-1.5 for stars and r^-1.8 for BHs in the relaxed inner region. These are not the same as assigning both populations the initial gamma or fitting a progenitor-selected histogram. The presets here describe initial populations, not that later Fokker-Planck equilibrium.

**Xue's zero radial index must not be inserted as spatial gamma=0.** That would create a completely different population. Also, a semimajor axis a is not an instantaneous spatial radius r for eccentric orbits.

### Pan: use the broken spatial density, not just gamma=1.5 everywhere

The implemented target is Pan Eq. (3):

$$
n(r)\propto r^{-\gamma}(r+r_a)^{\gamma-4},\qquad
F(<r)=\left(\frac{r}{r+r_a}\right)^{3-\gamma},
$$

$$
r_a=4r_h=\frac{4GM_\bullet}{\sigma^2},\qquad
\sigma=70\,\mathrm{km\,s^{-1}}
\left(\frac{M_\bullet}{1.53\times10^6M_\odot}\right)^{1/4.24}.
$$

The new sampler uses an isotropic Eddington inversion in the **Kepler** potential to recover this full spatial tracer shape. It does not incorrectly assign F(r) directly to a. In a Kepler potential, phase-volume weighting gives p(log E) proportional to f(E) E^-3/2, with a=GM/(2E), thermal eccentricity, isotropic cos(i), and uniform argument of periapsis.

At the reference SMBH mass, r_a=1.271607 pc. It follows the paper's M-sigma relation as M_bullet changes; it is not the legacy fixed NSC outer radius. The Pan profile has an extended tail rather than a hard truncation at that radius. Empirical `nsc_radius_*` inputs are unused by these presets and are identified as such in metadata.

The sampler is checked by drawing random orbital phases, reconstructing actual r, and comparing it to the analytic spatial CDF. This validates the radial-profile shape, not only the exponent of an a histogram.

**Remaining approximation:** Pan's stellar contribution to the gravitational potential and its subsequent Fokker-Planck/loss-cone evolution are not reproduced. Our dynamical engine remains Keplerian; self-gravity matters especially outside the influence region. Population alignment is not a claim of complete rate-model reproduction.

### Xue: explicit mass/radial proxy, not full kinematic reproduction

The radial range uses a source frequency of 1e-4 Hz for its inner boundary and r_BH,out=GM_bullet/sigma^2, sigma=265 km/s (M_bullet/1e8 Msun)^0.25 for the initial population outer radius. The implemented radial-law control draws uniform radius and represents it with **circular orbits, r=a**. Its orientations are isotropic; they do **not** reproduce Xue's beta_v=0.2 Gaussian velocity prescription. Accretion, hierarchical mergers, recoil and the published Kerr waveform are also absent. This preset is explicitly a mass/count/radial control, not a reproduction of Xue's event rates.

## Commands and outputs

```bash
# Explicit opt-in three-axis comparison: conditional Pan NSC normalization.
python scripts/compare_host_rates.py --population-model pan2021_nsc

# One-repeat population controls: keep all three under one output directory.
python scripts/compare_host_rates.py --population-model pan2021_nsc \
  --reference-only --repeats 1 --output-dir outputs/archive/populations/published_population_validation/pan2021_nsc
python scripts/compare_host_rates.py --population-model pan2021_paper \
  --reference-only --repeats 1 --output-dir outputs/archive/populations/published_population_validation/pan2021_paper
python scripts/compare_host_rates.py --population-model xue2026_bh \
  --reference-only --repeats 1 --output-dir outputs/archive/populations/published_population_validation/xue2026_bh

# Inspect normalization and verify the radial spatial CDF without a rate rerun.
python scripts/check_published_populations.py

# Reproduce an archived ad hoc IMF experiment explicitly.
python scripts/compare_host_rates.py --population-model legacy_popiii \
  --popiii-min-mass-msun 1 --popiii-max-mass-msun 400 \
  --output-dir outputs/archive/imf/imf_1_400_rates
```

`outputs/archive/populations/published_population_check/` contains the mass/number/crossing census, radial-CDF test, prescription metadata, combined pilot rate table (`reference_pilots.csv`) and report. The three pilot runs live together in `outputs/archive/populations/published_population_validation/`, one folder per model. Standard run metadata includes `population_model`, `population_prescription`, and per-grid component budgets. Cache checks distinguish population models; old IMF rates cannot be reused as published-population rates.

Reference pilots use one 10-Myr realization at z=10 and M_bullet=1e5 Msun. Their rates and interpretation are maintained in [results and uncertainties](simulation_results_synthesis.md#6-population-alternatives-are-not-an-uncertainty-distribution), not duplicated here. These controls change several assumptions together and are not converged reproductions of published rates. Detailed data live in `outputs/archive/populations/published_population_validation/`; the combined table is `outputs/archive/populations/published_population_check/reference_pilots.csv`.

## Verified sources

- Pan & Yang (2021), *Formation Rate of Extreme Mass Ratio Inspirals in Active Galactic Nuclei*, PRD 103, 103018, [DOI](https://doi.org/10.1103/PhysRevD.103.103018), arXiv:2101.09146v6: Eq. (3), Eq. (19), Sec. II.C.3. Fiducial gamma=1.5; gamma=1.8 is also explored.
- Xue et al. (2026), *Recoil-regulated extreme mass-ratio inspirals in AGN disks*, [arXiv:2605.29305v1](https://arxiv.org/abs/2605.29305), Sec. II.A and Table I. BH mass-law slope 2.35; gamma_rho=0 is the exponent of dN/dr.
