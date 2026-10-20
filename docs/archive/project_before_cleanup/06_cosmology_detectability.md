# Part 06 — Cosmology and Physical LISA Detectability

## 1. Idea

After gas delivery, the workflow keeps two deliberately separate models:

1. the existing astrophysical redshift-time and simplified central-SMBH merger
   bookkeeping used to decide whether an intrinsic candidate reaches the
   modeled entry boundary;
2. first-order physical LISA detector selection using luminosity distance, an
   analytic LISA strain PSD, a circular inspiral waveform, and a finite mission
   interval.

The detector result is a sky/orientation-averaged inspiral SNR conditional on
the modeled source state. It is not a complete LISA IMRI selection function and
does not turn the per-host intrinsic yield into a cosmological detection rate.

## 2. Astrophysical time and frequency bookkeeping

`src/workflows/redshift_evolve.py::RedshiftEvolution` remains the owner of the
existing event-time trajectory,

```text
(1 + z) = (H0 t)^(-2/3),
```

with `hubble_constant_per_myr` in `1/Myr`. This model advances redshift across
gas time, Peters evolution, and the optional simplified cosmic merger. It is
not used for luminosity distance.

`src/workflows/cosmic_merge.py::CosmicMergeModel` may apply one event before
entry, multiplying the SMBH mass and radius by the configured factors. Source
and observer frequencies retain

```text
f_source = (1 + z) f_observer.
```

## 3. Detector calculation

`src/workflows/lisa_snr.py` owns the reusable detector physics and
`src/workflows/detect_rate.py::classify_detectability` owns workflow
classification.

For source-frame component masses, the detector kernel computes

```text
Mchirp_source = (m1 m2)^(3/5) / (m1 + m2)^(1/5)
Mchirp_detector = (1 + z) Mchirp_source
D_L = (1 + z) c/H0 integral_0^z dz'/sqrt(Omega_m(1+z')^3 + 1-Omega_m)
f_ISCO,obs = c^3 / [6^(3/2) pi G (m1+m2) (1+z)]
```

The detector distance defaults are `H0=67.4 km/s/Mpc` and `Omega_m=0.315`.
They are distinct from `hubble_constant_per_myr`. Radiation is omitted from
this stated flat matter-plus-Lambda approximation.

The two-channel Robson–Cornish–Liu analytic model supplies an effective strain
noise PSD `S_n(f)` in `Hz^-1`, including 2.5 Gm arms, optical-metrology noise,
acceleration noise, finite-arm response, and the mission-duration Galactic
foreground. The SNR never consumes ASD.

For the circular, nonspinning, dominant-harmonic SPA amplitude,

```text
A(f) = sqrt(5/24) (G Mchirp_detector/c^3)^(5/6)
       / [pi^(2/3) (D_L/c)] f^(-7/6)
```

the RCL sky/polarization effective PSD and binary-inclination average give

```text
rho^2 = (16/5) integral A(f)^2 / S_n(f) df.
```

The mission interval begins at the actual observer-frame source frequency. It
can describe a source already in band, entering during the mission, reaching
ISCO, reaching the detector upper bound, evolving only over a mission-limited
track, or never overlapping. A physical no-overlap has SNR zero. Missing input,
unsupported eccentricity, and numerical/model failure remain unavailable.

## 4. Classification and compatibility

Finite SNR is compared with `snr_threshold` and yields
`lisa_snr_above_threshold` or `lisa_snr_below_threshold`. No overlap,
unavailable input, model unavailability, and numerical failure have distinct
statuses and reasons. Non-candidates are detector `not evaluated` rather than
below threshold.

`lisa_snr` is authoritative. `snr_proxy` remains in the event schema only as a
deprecated exact alias of `lisa_snr`; the old threshold-valued heuristic is no
longer executed.

Detector-evaluated and detector-unavailable candidate counts are persisted
separately. Detector unavailability makes the detector-selected rate
unavailable without invalidating an otherwise complete intrinsic IMRI rate.

## 5. Key configuration

| Parameter | Default | Meaning |
|---|---:|---|
| `redshift` | 15 | initial astrophysical source redshift |
| `hubble_constant_per_myr` | `1/13.6e3` | existing event-time scale |
| `lisa_min_gw_frequency_hz` | `1e-3` | retained intrinsic entry and detector lower bound |
| `lisa_max_gw_frequency_hz` | `1e-1` | retained detector upper bound |
| `lisa_mission_duration_years` | 4 | observer mission and RCL foreground duration |
| `snr_threshold` | 20 | authoritative finite-SNR threshold |
| `detector_hubble_constant_km_s_mpc` | 67.4 | detector luminosity-distance H0 |
| `detector_omega_m` | 0.315 | detector flat-LambdaCDM matter density |

## 6. Approximation boundary

This is first-order physical LISA inspiral detectability for circular,
nonspinning IMRIs, matching the current Monte Carlo assumptions. It omits
relativistic IMRI waveforms, higher harmonics, merger/ringdown,
source-orientation sampling, full orbital/TDI response, lensing, and
mission-calendar effects. The next priority is to quantify population-level
Monte Carlo variation, threshold-edge events, and the conditional-rate versus
cosmological-rate boundary. Eccentricity or spin should be added only together
with corresponding source-population variables and evidence that they are the
dominant limitation.
