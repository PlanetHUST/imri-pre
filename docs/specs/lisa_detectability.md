# LISA Detectability Specification

Status: implemented authoritative specification

Implementation status: integrated and validated on 2026-09-02

Last repository review: 2026-09-02

## 1. Scientific objective

Replace the current threshold-valued `snr_proxy` fallback with a first-order
physical LISA inspiral signal-to-noise ratio (SNR). The result must support
scientifically meaningful comparisons across source mass and redshift, while
remaining explicit about the approximation boundary for IMRI sources.

The authoritative detector statistic will be `lisa_snr`. It will be computed
from a frequency-domain inspiral amplitude, luminosity distance, the
sky-averaged LISA effective strain power spectral density (PSD), and a finite
observer-frame mission window. It must not be inferred from band membership or
set equal to the configured threshold.

This upgrade produces a conditional, first-order detector selection for an
intrinsic event that reaches the modeled LISA band. It does not by itself
convert the per-host, per-AGN-episode IMRI yield into a cosmological detection
rate.

## 2. Scope and invariants

### 2.1 Intrinsic formation and evolution

The intrinsic layer decides whether an embedded object becomes an IMRI
candidate and how it reaches the existing observed-frequency entry boundary.
This layer includes population sampling, damping, compact-object formation,
disk migration, gap selection, traps, gas-to-GW handoff, Peters transport, the
AGN duty cycle, and band-entry/ISCO geometry.

The detector upgrade must not change those calculations, their thresholds, or
their status meanings. It must not change the disk model, torque profile, gap
criterion, trap behavior, migration radii, duty cycle, Peters transport,
candidate definition, paired-seed policy, or intrinsic-rate denominator.

The current `lisa_min_gw_frequency_hz` value is also the intrinsic event-entry
boundary. Changing that default would change entry radii, migration routes, and
intrinsic candidate rates. Therefore this task must not silently change the
existing `1e-3 Hz` lower entry boundary or `1e-1 Hz` upper bound. A future
change to the physical band definition is a separate paired-seed science task.

### 2.2 Detector selection

The detector layer begins only after the intrinsic layer has supplied an IMRI
candidate and its post-gas state. It owns detector-frame mass and frequency
quantities, luminosity distance, LISA sensitivity, waveform amplitude,
mission-limited integration, `lisa_snr`, threshold classification, and
detector-specific statuses.

Intrinsic candidate counts and rates must remain available when detector SNR
is unavailable. Detector-model unavailability or numerical failure must make
the detector-selected result unavailable, not turn an intrinsic event into
`no-IMRI` and not create a numeric zero rate.

## 3. Pre-implementation baseline (historical)

This section records the repository state inspected on 2026-09-02.

### 3.1 Current call path

```text
workflow.run_simulation
  -> IMRISingleWorkflow.run_single_simulation
  -> IMRIEventEvaluator.classify_imri_candidates
  -> circular_band_boundary
  -> resolve_band_entry_route
  -> classify_event_rate
  -> classify_detectability
  -> IMRIEventRecord and per-repeat counters
  -> outputs.py CSV/JSON persistence
```

`src/workflows/event_work.py` is the integration owner. It first calls
`physical_process.gw_frequency_radius.circular_band_boundary()` with the
configured initial redshift and lower observed frequency. An entry at or below
Schwarzschild ISCO is routed to `classify_unsupported_band_entry()` and
`unsupported_band_entry_decision()` before a migration profile or detector SNR
is evaluated.

For a valid boundary, `event_work.py` resolves an in-domain entry or a
gas-to-GW handoff, calls `classify_event_rate()`, and then passes the resulting
`is_imri_candidate`, `r_gas_fin_cm`, and `gas_final_time_myr` to
`classify_detectability()`.

### 3.2 Current detectability behavior

`src/workflows/detect_rate.py` currently:

1. returns `non-detectable`, `False`, and `snr_proxy=0` for a non-candidate;
2. evolves the initial redshift to the gas-final time with
   `RedshiftEvolution`;
3. optionally applies one simplified central-SMBH merger before entry;
4. uses Peters evolution to the configured lower observed frequency;
5. rejects paths that exceed the current remaining-cosmic-time model;
6. recomputes the entry radius and frequency at the detection redshift;
7. calls `single_frequency_snr_proxy()`.

The proxy is exactly

```text
snr_proxy = snr_threshold  when f_min <= f_obs <= f_max
snr_proxy = 0              otherwise
```

The default threshold is `20`. Every in-band positive result therefore has the
same stored value and cannot encode distance, source strength, LISA noise,
mission duration, or a ranking among sources.

Current raw detector classes are `imri data`, `non-detectable`, and
`not evaluated`. Current reasons include `not_imri_candidate`,
`invalid_gas_final_radius`, `invalid_frequency`, `gw_evolution_unavailable`,
`misses_cosmic_time_before_merge`, `misses_cosmic_time_budget`,
`outside_lisa_band`, and `band_entry_inside_isco`.

### 3.3 Current redshift model

`src/workflows/redshift_evolve.py` uses

\[
1+z=(H_0 t)^{-2/3},
\]

with analytic forward and inverse time relations. It is not a flat-
\(\Lambda\)CDM luminosity-distance implementation. Before implementation, the
detector upgrade must resolve whether this event-time trajectory is retained
or replaced inside the detector stage by the same flat-\(\Lambda\)CDM object
used for distance. Mixing two cosmologies without explicit provenance is not
acceptable.

### 3.4 Current data and output ownership

- `DetectRateDecision` contains `snr_proxy`, entry frequencies, detection
  time/redshift, final SMBH mass/radius, GW time, merger flag, class, boolean,
  and reason.
- `IMRIEventRecord` repeats those detector fields in the typed event schema.
- `event_work.py` copies the decision into the event record and increments
  detector counters only for intrinsic candidates.
- `IMRIDynSingleResult` owns detectable and non-detectable counts, masses, and
  per-Myr rates.
- `IMRIDynAveragedResult` averages those fields only when the existing
  band-entry/profile completeness gate is satisfied.
- `outputs.py` owns explicit CSV ordering and save-time unit formatting.
  `snr_proxy` is written to `imri_events.csv`; it is absent from repeat and
  averaged tables.
- `final_analysis.py` consumes `is_lisa_detectable`, not `snr_proxy`.
- `redshift_scan.py` carries `snr_threshold` through a strict configuration
  template and combines per-run events and repeat tables.
- `metadata.py` persists the threshold and band bounds but no waveform,
  distance, mission, sensitivity, or integration provenance.

There is no focused detectability or LISA-sensitivity validation harness. The
visualization harness tests synthetic detector labels and preservation of
`not evaluated`; it does not validate SNR physics.

## 4. Target detector model

The first implementation will use:

- the Robson, Cornish & Liu (2019) analytic two-channel, sky-averaged LISA
  sensitivity curve;
- arm length \(L=2.5\times10^9\,\mathrm{m}\);
- optical-metrology and test-mass acceleration noise;
- the analytic finite-arm response fit;
- the mission-duration-dependent unresolved Galactic-binary foreground;
- flat-\(\Lambda\)CDM luminosity distance;
- a quasi-circular, nonspinning, dominant-quadrupole, inspiral-only waveform;
- sky, polarization, and binary-inclination averaging;
- a finite observer-frame mission duration;
- numerical integration of SNR over observed GW frequency.

The sensitivity and waveform kernels should be independently testable physical
helpers. `detect_rate.py` remains the workflow classification owner; it should
not become the sole owner of all detector numerics.

## 5. Mathematical conventions

All equations in this section are implementation requirements unless marked as
an unresolved decision.

### 5.1 Source masses and chirp mass

Let \(m_1\) and \(m_2\) be source-frame component masses, and

\[
M=m_1+m_2,
\qquad
\eta=\frac{m_1m_2}{M^2}.
\]

The source-frame chirp mass is

\[
\mathcal{M}=\frac{(m_1m_2)^{3/5}}{M^{1/5}}=M\eta^{3/5}.
\]

At detector-evaluation redshift \(z\),

\[
M_z=(1+z)M,
\qquad
\mathcal{M}_z=(1+z)\mathcal{M}.
\]

Persisted component masses remain source-frame masses. Redshifted masses are
derived detector quantities and must not be passed into source-frame
orbital-radius helpers.

### 5.2 Frequency transformation

The existing convention is retained:

\[
f_{\rm src}=(1+z)f_{\rm obs},
\qquad
f_{\rm obs}=\frac{f_{\rm src}}{1+z}.
\]

All LISA PSD and SNR frequencies are observed-frame frequencies.
`gw_frequency_hz()` and source orbital radii continue to use source-frame total
mass and source-frame frequency. A frequency or mass may be redshifted once,
never both before and inside the same helper.

### 5.3 Flat-LambdaCDM luminosity distance

For a flat matter-plus-cosmological-constant model,

\[
E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda},
\qquad
\Omega_\Lambda=1-\Omega_m,
\]

\[
D_L(z)=(1+z)\frac{c}{H_0}\int_0^z\frac{dz'}{E(z')}.
\]

The implementation uses explicit \(H_0\) in `km s^-1 Mpc^-1`, returns
`luminosity_distance_mpc`, and records \(H_0\), \(\Omega_m\), flatness,
radiation treatment, and integration method. The proposed Planck-2018 baseline
is

\[
H_0=67.4\ {\rm km\,s^{-1}\,Mpc^{-1}},
\qquad
\Omega_m=0.315.
\]

Radiation is omitted in this matter-plus-Lambda expression unless the
pre-implementation cosmology decision adopts a shared model that includes it.
That choice must be explicit because scans extend to high redshift.

### 5.4 Observed Schwarzschild-ISCO frequency

\[
f_{\rm ISCO,src}=\frac{c^3}{6^{3/2}\pi G M},
\]

\[
f_{\rm ISCO,obs}
=\frac{f_{\rm ISCO,src}}{1+z}
=\frac{c^3}{6^{3/2}\pi G M_z}.
\]

This is an inspiral termination convention, not a merger/ringdown model.

### 5.5 Fourier-domain inspiral amplitude

For observed frequency \(f\), use

\[
A(f)=\sqrt{\frac{5}{24}}
\frac{(G\mathcal{M}_z/c^3)^{5/6}}
     {\pi^{2/3}(D_L/c)}f^{-7/6}.
\]

The polarization amplitudes follow the Robson et al. convention:

\[
\tilde h_+(f)=A(f)\frac{1+\cos^2\iota}{2}e^{i\Psi(f)},
\qquad
\tilde h_\times(f)=iA(f)\cos\iota\,e^{i\Psi(f)}.
\]

### 5.6 LISA effective strain PSD

Let

\[
L=2.5\times10^9\ {\rm m},
\qquad
f_*=\frac{c}{2\pi L}.
\]

The single-link noise terms are

\[
P_{\rm OMS}(f)=
(1.5\times10^{-11}\ {\rm m})^2
\left[1+\left(\frac{2\times10^{-3}\ {\rm Hz}}{f}\right)^4\right]
\ {\rm Hz}^{-1},
\]

\[
P_{\rm acc}(f)=
(3\times10^{-15}\ {\rm m\,s^{-2}})^2
\left[1+\left(\frac{0.4\times10^{-3}\ {\rm Hz}}{f}\right)^2\right]
\left[1+\left(\frac{f}{8\times10^{-3}\ {\rm Hz}}\right)^4\right]
\ {\rm Hz}^{-1}.
\]

The two-channel effective instrumental strain PSD is

\[
S_{\rm inst}(f)=\frac{10}{3L^2}
\left[P_{\rm OMS}(f)
+2\left(1+\cos^2(f/f_*)\right)
\frac{P_{\rm acc}(f)}{(2\pi f)^4}\right]
\left[1+0.6\left(\frac{f}{f_*}\right)^2\right].
\]

The unresolved Galactic foreground is

\[
S_c(f)=Af^{-7/3}
\exp\left[-f^\alpha+\beta f\sin(\kappa f)\right]
\left[1+\tanh\left(\gamma(f_k-f)\right)\right]
\ {\rm Hz}^{-1},
\]

with \(A=9\times10^{-45}\) and frequency supplied in Hz. Published parameter
sets are:

| Mission duration | alpha | beta | kappa | gamma | `f_k_hz` |
|---:|---:|---:|---:|---:|---:|
| 0.5 yr | 0.133 | 243 | 482 | 917 | 0.00258 |
| 1 yr | 0.171 | 292 | 1020 | 1680 | 0.00215 |
| 2 yr | 0.165 | 299 | 611 | 1340 | 0.00173 |
| 4 yr | 0.138 | -221 | 521 | 1680 | 0.00113 |

The total effective strain PSD is

\[
S_n(f)=S_{\rm inst}(f)+S_c(f).
\]

`S_n` is a one-sided PSD with units `Hz^-1`. The amplitude spectral density
(ASD) is \(\sqrt{S_n}\), with units `Hz^-1/2`. Code and documentation must not
call an ASD a PSD or square an already-squared sensitivity.

No interpolation among the four published foreground fits is permitted in the
first implementation. An unsupported mission duration fails explicitly.

### 5.7 Inspiral time-frequency relation

The observer-frame time remaining to formal coalescence is

\[
\tau(f)=\frac{5}{256}
\left(\frac{G\mathcal{M}_z}{c^3}\right)^{-5/3}
(\pi f)^{-8/3}.
\]

Its inverse is

\[
f(\tau)=\frac{1}{\pi}
\left(\frac{5}{256\tau}\right)^{3/8}
\left(\frac{G\mathcal{M}_z}{c^3}\right)^{-5/8}.
\]

Both quantities are detector-frame here. Mission duration is observer-frame
years and is not divided by \(1+z\) again.

### 5.8 Orientation-averaged SNR

The chosen effective PSD already contains the sky/polarization-averaged LISA
response. Averaging the polarizations above over binary inclination gives

\[
\left\langle |\tilde h|^2\right\rangle
=\frac{4}{5}\mathcal{R}(f)A^2(f).
\]

With \(S_n=P_n/\mathcal{R}\), the required convention is

\[
\boxed{\overline{\rho^2}
=\frac{16}{5}\int_{f_{\rm low}}^{f_{\rm high}}
\frac{A^2(f)}{S_n(f)}\,df}.
\]

The `16/5` factor is fixed by the Robson et al. convention paired with the
amplitude and effective PSD above. Using `4 integral A^2/S_n df` would omit the
chosen inclination convention; applying another detector response would
double-count sky/polarization averaging.

## 6. Observation-window semantics

### 6.1 First-order window definition

The repository has no LISA mission start epoch or event-to-calendar overlap
model. The first calculation is therefore a source-aligned conditional SNR:
the best available interval of at most the configured mission duration ending
at modeled inspiral termination. It does not estimate the probability that the
event overlaps the actual mission.

Define

\[
f_{\rm upper}=\min(f_{\rm detector,max},f_{\rm ISCO,obs}).
\]

For observer-frame duration \(T_{\rm obs}\), define

\[
f_T=f\left(\tau(f_{\rm upper})+T_{\rm obs}\right).
\]

The available source-state lower frequency is

\[
f_{\rm available}=\max(f_{\rm detector,min},f_{\rm current,obs}),
\]

where `f_current_obs` is the actual frequency when detector evaluation becomes
applicable. It must come from the retained physical radius; it must not be
reset outward to formal entry when a source is already inside the band.

Then

\[
f_{\rm lower}=\max(f_T,f_{\rm available}).
\]

### 6.2 Required cases

- Clip to configured observed detector bounds; do not silently extend them.
- Always truncate the inspiral at observed ISCO.
- Preserve the upstream entry contract. `band_entry_inside_isco` remains
  `not evaluated`; the SNR kernel does not reinterpret it as zero.
- For a source already in band, begin at its actual current frequency and do
  not integrate signal history that has passed.
- For a source entering within the source-aligned window, begin at its
  entry/available frequency, yielding a duration shorter than the mission.
- For a source in band longer than the mission, begin at `f_T`.
- For a source reaching ISCO within the mission, stop at ISCO.
- If the configured upper bound is below ISCO, stop there and invert relative
  to that bound.
- If `f_lower >= f_upper`, return evaluated `lisa_snr=0`, status
  `no_lisa_frequency_overlap`, and `is_lisa_detectable=False`. This is a
  physical result, not a numerical failure.

Mission-calendar overlap remains out of scope. A future rate model must add
mission dates or overlap probability rather than treating conditional SNR as a
calendar selection probability.

## 7. Redshift conventions and double-redshift prevention

- Component masses and persisted event masses are source-frame masses.
- Source orbital radii and `gw_frequency_hz()` use source-frame total mass.
- `detection_redshift` is the detector-evaluation redshift after modeled delays.
- `luminosity_distance_mpc` uses `detection_redshift`, not formation redshift.
- Detector total and chirp masses are derived exactly once.
- Observed waveform/time-frequency equations use redshifted chirp mass and
  observed frequency.
- ISCO computed by source-frequency redshifting must agree with the
  detector-mass expression.
- Helpers that convert `f_obs` to `f_src` must never receive an already
  redshifted frequency.
- Detector-frame mass must never enter a helper that also applies redshift.

## 8. Status and classification model

`lisa_snr` is authoritative only for an evaluated physical outcome. A new raw
detector status must distinguish at least:

| Raw status | `lisa_snr` | `is_lisa_detectable` | Meaning |
|---|---:|---:|---|
| `not_applicable_not_imri_candidate` | unavailable | `None` preferred | No intrinsic event enters the detector denominator |
| `not_evaluated_upstream` | unavailable | `None` | Upstream geometry/candidacy was unavailable |
| `snr_model_input_unavailable` | unavailable | `None` | Required distance, mass, frequency, or supported waveform input is absent |
| `snr_numerical_failure` | unavailable | `None` | Finite valid input failed numerical evaluation |
| `no_lisa_frequency_overlap` | `0.0` | `False` | Physical evaluated zero overlap |
| `lisa_snr_below_threshold` | finite nonnegative | `False` | Evaluated below threshold |
| `lisa_snr_at_or_above_threshold` | finite nonnegative | `True` | Evaluated at or above threshold |

The existing presentation classes may remain for compatibility:

- `imri data` maps only from `lisa_snr_at_or_above_threshold`;
- `non-detectable` maps only from evaluated false outcomes;
- `not evaluated` maps from upstream, unavailable, and failure outcomes, with
  raw status/reason retaining their distinction.

The existing `not_imri_candidate -> non-detectable/False` behavior is not a
sound detector-denominator statement. Whether it changes to not-applicable or
is retained temporarily is an explicit schema decision. Non-candidates must
not increment the non-detectable IMRI counter.

Numerical exceptions must become `snr_numerical_failure`, not zeros. Likewise,
the circular waveform must not silently evaluate a configured nonzero
eccentricity. Until an eccentric waveform exists, that event is model-input
unavailable or configuration-invalid, according to the recorded decision.

## 9. Configuration and provenance

| Parameter | Unit/type | Proposed default | Requirement |
|---|---|---:|---|
| `lisa_min_gw_frequency_hz` | Hz | `1e-3` | Existing entry and lower integration bound; unchanged here |
| `lisa_max_gw_frequency_hz` | Hz | `1e-1` | Existing upper bound; unchanged here |
| `lisa_mission_duration_years` | observer yr | `4.0` | Must select a published foreground fit |
| `snr_threshold` | dimensionless | `20.0` | Applied only to finite authoritative SNR |
| `detector_hubble_constant_km_s_mpc` | km s^-1 Mpc^-1 | `67.4` | Positive flat-LambdaCDM parameter |
| `detector_omega_m` | dimensionless | `0.315` | Baseline requires `0 < omega_m < 1` |
| `lisa_sensitivity_model` | identifier | `robson_cornish_liu_2019` | Versioned provenance |
| `lisa_confusion_noise_model` | identifier | `robson_cornish_liu_2019` | Includes selected duration |
| `lisa_waveform_model` | identifier | `spa_newtonian_circular_inspiral` | Explicit approximation |
| `lisa_snr_averaging` | identifier | `sky_polarization_inclination_averaged` | Fixes the `16/5` convention |

Metadata must also record fixed arm length, PSD equation/version, quadrature
method and resolution/tolerance, ISCO convention, and radiation treatment.

The existing `hubble_constant_per_myr` belongs to `RedshiftEvolution` and must
not be silently reinterpreted as `km s^-1 Mpc^-1`. If the detector-time model
is upgraded, deprecation/migration of that parameter must be explicit.

`scripts/redshift_scan.py` has a strict required-field list. Every new detector
parameter must enter its default template, strict field list, compatibility
policy, and scan metadata. Historical metadata without new physics parameters
must not receive silent inferred values.

## 10. Required interfaces and output fields

### 10.1 Detector-kernel result

A typed detector-kernel result should carry:

- `lisa_snr` and raw `lisa_snr_status`;
- `luminosity_distance_mpc`;
- source and detector chirp masses in `msun`;
- `snr_frequency_min_hz` and `snr_frequency_max_hz`;
- `observed_isco_frequency_hz`;
- `effective_observation_time_yr`;
- sensitivity, foreground, waveform, cosmology, and averaging identifiers;
- a bounded reason/message when unavailable or failed.

### 10.2 Event decision and event schema

`DetectRateDecision` and `IMRIEventRecord` must expose authoritative SNR,
status, distance, integration bounds, observed ISCO, effective duration, and
the existing detector class/boolean/reason. `snr_proxy` is present only if the
compatibility decision retains it.

If retained, `snr_proxy` must equal `lisa_snr` for evaluated events and share
its unavailable representation otherwise. It must never retain threshold-
valued behavior. Metadata must mark it deprecated.

### 10.3 Counters and rates

The result model needs detector-evaluated, detector-unavailable, numerical-
failure, no-overlap, below-threshold, and above-threshold accounting, or an
equivalent conserved taxonomy. Intrinsic and detector completeness must be
separable: a detector failure may invalidate `rate_det_per_myr` without
invalidating `rate_imri_per_myr`.

Existing compact `n_det`, `n_nondet`, and detector-rate outputs may remain, but
their denominators must be reconstructible. Unavailable persists as NaN/JSON
`null`, never zero.

### 10.4 Persistence

`outputs.py` remains the presentation owner. New frequency fields use its
save-time unit mechanism if shown in the event CSV. Canonical in-memory names
retain units. Metadata and per-repeat JSON remain strict with `allow_nan=False`
and explicit statuses.

## 11. Validation requirements

A focused standalone harness, proposed as
`scripts/lisa_detectability_test.py`, must cover the following.

### 11.1 LISA PSD

- Reproduce the RCL equations independently enough to catch missing factors.
- For the 4-year curve at `1e-3 Hz`, verify a pinned benchmark. The equations
  above give approximately `8.30e-38 Hz^-1`, or ASD
  `2.88e-19 Hz^-1/2`; finalize tolerance after independent review.
- Verify positive finite PSD across the configured band.
- Verify nonnegative instrument and foreground components and their sum.
- Verify every supported duration row and explicit rejection of others.

### 11.2 Cosmology

- Benchmark `D_L` at `z=0.1`, `1`, `10`, and `15` against an independent
  reference with identical parameters.
- Verify `D_L(0)=0`, monotonicity, units, and invalid inputs.
- If one cosmology owns time and distance, test `z -> t -> z` and remaining
  time.

### 11.3 Mass and redshift

- Verify chirp-mass identities and single redshifting.
- Verify source/observed frequency round trips.
- Verify both observed-ISCO paths agree.
- Include a regression that fails under double redshifting.

### 11.4 Mission window

- Verify the time-frequency inverse and full-duration difference.
- Test already-in-band, entering-during-window, ISCO-before-end,
  upper-bound-limited, and no-overlap cases.
- Verify observed-frame, ordered, finite integration bounds.

### 11.5 SNR and classification

- Verify inverse-distance SNR scaling for fixed detector-frame parameters.
- Verify nonnegative integrands and numerical convergence.
- Test representative IMRIs at low redshift and `z=10`, `13`, `15`.
- Test above/below threshold, no overlap, unavailable input, numerical failure,
  upstream not-evaluated, and non-candidate paths.
- Verify unavailable/failure never becomes a physical zero.
- If retained, assert `snr_proxy == lisa_snr` and absence of old proxy behavior.

### 11.6 Workflow and persistence

- Run a deterministic one-point workflow smoke with a persisted base seed.
- Check event/repeat/averaged headers and units.
- Check strict JSON maps nonfinite diagnostics to `null` with status/reason.
- Verify detector status counts conserve their denominator.
- In a paired-seed before/after run, verify intrinsic rates/outcomes remain
  unchanged while detector-selected results may change.
- Run the visualization harness to preserve `not evaluated` semantics.

## 12. Scientific limitations

This model is a material improvement over band membership, but not a complete
IMRI/EMRI LISA selection function. It omits:

- eccentric harmonics and eccentric waveform evolution;
- relativistic IMRI/EMRI waveform structure and self-force effects;
- spins and spin-dependent ISCO;
- higher harmonics;
- merger and ringdown;
- individual orientation and sky sampling;
- full time-dependent LISA orbital/TDI response;
- lensing and magnification scatter;
- mission duty cycle, gaps, and calendar overlap;
- a probabilistic `p_det` marginalized over missing variables.

Robson et al. note that EMRIs and other complicated sources require more
involved waveform treatment. Results must be labeled “sky-averaged circular-
inspiral SNR,” not a final LISA detection probability.

## 13. Fixed scientific and architectural decisions

1. Intrinsic formation and existing entry-frequency defaults do not change.
2. `lisa_snr` is authoritative; threshold-valued SNR is removed.
3. The first waveform is circular, nonspinning, quadrupole, and inspiral-only.
4. The RCL two-channel PSD and `16/5` convention are used together.
5. PSD/SNR frequencies are observed-frame frequencies.
6. Source masses remain canonical; redshifted masses are derived once.
7. SNR is source-aligned and conditional because no mission calendar exists.
8. No overlap is evaluated zero; unavailable input and failure are unavailable.
9. Detector and intrinsic completeness are separate.
10. Detector numerics are independently testable helpers; `detect_rate.py`
    remains the classifier.

## 14. Resolved implementation decisions

1. The existing `RedshiftEvolution` remains the owner of the astrophysical
   event-time trajectory. Flat-LambdaCDM is a separate detector-distance model;
   the two parameter sets have distinct names and metadata provenance.
2. Detector distance uses the requested flat matter-plus-Lambda expression;
   radiation is explicitly omitted and recorded as such.
3. Nonzero configured eccentricity leaves the intrinsic event intact but makes
   detector SNR unavailable with
   `snr_model_unavailable_nonzero_eccentricity`.
4. A non-candidate is detector `not evaluated` with boolean `None`, not a
   below-threshold source.
5. `snr_proxy` is retained for compatibility only and always aliases
   `lisa_snr`; the heuristic formula was removed.
6. Cosmological distance uses adaptive `scipy.integrate.quad` with relative
   tolerance `1e-11`. SNR uses the same integrator in log observed frequency
   with relative tolerance `1e-8`.

## 15. Implemented interfaces and validation record

The reusable detector kernel is `src/workflows/lisa_snr.py`. The standalone
reference and regression harness is `scripts/lisa_detectability_test.py`.
`detect_rate.py` calls that same kernel after the existing gas, Peters,
redshift-time, and optional cosmic-merger stages.

For the reference source

```text
m1_source_msun = 100000
m2_source_msun = 100
z = 1
current f_obs = 0.001 Hz
mission = 4 yr
```

the implemented calculation gives:

```text
chirp_mass_source_msun       = 1584.5764038704444
chirp_mass_detector_msun     = 3169.152807740889
luminosity_distance_mpc      = 6802.525834693799
f_isco_source_hz             = 0.04392649145417517
f_isco_observer_hz           = 0.021963245727087584
f_low_hz                     = 0.001
f_high_hz                    = 0.021963245727087584
observation_duration_years   = 2.9975644979861324
observation_case             = already_in_band:reaches_isco_during_mission
S_n(0.001 Hz)                = 8.296514847664091e-38 Hz^-1
sqrt(S_n(0.001 Hz))          = 2.8803671376517423e-19 Hz^-1/2
A(0.001 Hz)                  = 3.0008960779408375e-17 Hz^-1
lisa_snr                     = 58.07069322689072
classification               = lisa_snr_above_threshold
```

Adaptive quadrature and a 200001-point log-frequency trapezoid agree to better
than `2e-9` relative. Holding detector-frame chirp mass and frequency bounds
fixed gives exactly the expected `rho(D)/rho(2D)=2` scaling to numerical
precision. Flat-LambdaCDM distances in Mpc are `477.5220941`, `6802.525835`,
`47676.04091`, `105971.0815`, and `167196.7063` at redshifts `0.1`, `1`,
`5`, `10`, and `15`.

The controlled `m1=(1e4,1e5,1e6,1e7) Msun`,
`m2=(10,100,1000) Msun`, `z=(0.1,0.5,1,2,5,10,13,15)` grid produced only
evaluated or physical-no-overlap results. In particular, `1e7 Msun` systems
have observed Schwarzschild ISCO below the retained `1e-3 Hz` lower boundary
across that redshift grid. The non-monotonic SNR tracks near the
mission-duration transition are physical consequences of redshifted chirp
mass and finite frequency coverage, not corrections to inverse-distance
scaling.

## 16. References

1. T. Robson, N. J. Cornish, and C. Liu, “The construction and use of LISA
   sensitivity curves,” *Classical and Quantum Gravity* 36, 105011 (2019),
   [arXiv:1803.01944](https://arxiv.org/abs/1803.01944),
   [doi:10.1088/1361-6382/ab1101](https://doi.org/10.1088/1361-6382/ab1101).
   Supports the effective PSD, noise/foreground fits, averaging convention,
   inspiral amplitude, SNR integral, and time-frequency relation.
2. S. Babak, M. Hewitson, and A. Petiteau, “LISA Sensitivity and SNR
   Calculations,” LISA-LCST-SGS-TN-001 (2021),
   [arXiv:2108.01167](https://arxiv.org/abs/2108.01167). Supports one-sided PSD,
   response, sky-averaged sensitivity, TDI context, and SNR conventions.
3. M. Colpi et al., “LISA Definition Study Report” (2024),
   [arXiv:2402.07571](https://arxiv.org/abs/2402.07571). Supports the adopted
   design context, mission band, and approximation boundary.
4. Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,”
   *Astronomy & Astrophysics* 641, A6 (2020),
   [arXiv:1807.06209](https://arxiv.org/abs/1807.06209). Supports proposed
   `H0=67.4 km s^-1 Mpc^-1` and `Omega_m=0.315`.
5. P. C. Peters, “Gravitational Radiation and the Motion of Two Point Masses,”
   *Physical Review* 136, B1224 (1964),
   [doi:10.1103/PhysRev.136.B1224](https://doi.org/10.1103/PhysRev.136.B1224).
   Supports the existing vacuum-GW framework and leading circular evolution.
6. C. J. Moore, R. H. Cole, and C. P. L. Berry, “Gravitational-wave sensitivity
   curves,” *Classical and Quantum Gravity* 32, 015014 (2015),
   [arXiv:1408.0740](https://arxiv.org/abs/1408.0740). Supports PSD, ASD, and
   characteristic-strain distinctions.
