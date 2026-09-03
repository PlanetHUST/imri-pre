# Research Roadmap: Local IMRI Response to Cosmological Rates

This document condenses the durable future-modeling guidance from the former
project guide and merger-tree analysis. Current code implements a local
per-host response; the roadmap explains how to extend it without confusing
local yield, population weighting, and detector selection.

## 1. Current kernel and missing bridge

The present workflow answers:

```text
given host/scenario parameters lambda,
how many candidates form during one AGN episode,
and which candidates pass the current conditional detector model?
```

It does not supply the cosmological abundance of hosts, their ancestry,
occupation fraction, AGN duty history, or a volumetric observer rate. The
missing bridge is an event-driven carrier model joining merger-tree histories
to a reusable local response library.

## 2. Keep physical channels explicit

At minimum, label separately:

- stellar-mass BH delivery through the central AGN disk;
- cluster-carried compact binaries or IMBH/SBH systems;
- satellite/galaxy-carried systems;
- dual-SMBH/nuclear interactions;
- disrupted, ejected, recoiled, merged, or otherwise terminated systems.

Do not merge name-similar channels or count the same ancestral object after it
changes carrier. Formation, transport, intrinsic coalescence, and detection are
separate events.

## 3. State and clocks

Propagate two linked states.

### Internal compact-binary/source state

```text
component masses and source frame
semi-major axis/eccentricity or equivalent orbital state
formation and evolution timestamps
channel and physical-status labels
intrinsic/event/detector outcomes
```

### External carrier/environment state

```text
halo, galaxy, nuclear cluster, or free/ejected carrier identity
host mass and redshift
central-BH and NSC properties
AGN state and disk/scenario parameters
orbital position/velocity or reduced transport state
stripping, disruption, recoil, and merger history
```

Ancestry IDs are immutable. Carrier IDs may change. Record the event that made
the change rather than overwriting history.

Keep three clocks explicit:

1. cosmological/tree time;
2. carrier transport/survival time;
3. local source evolution time.

Convert between time and redshift through one stated cosmology. Do not use a
detector luminosity-distance cosmology as an unstated replacement for the
existing local event-time approximation.

## 4. Environmental decision diagnostics

Use these as routing diagnostics or calibration targets, not universal kernels:

| Question | Compact diagnostic | Interpretation |
|---|---|---|
| Does the environment act before merger? | `eta_env = t_GW/t_next,env` | `eta_env << 1`: GW-first; `eta_env >= 1`: follow environment explicitly |
| Can a smooth external tide perturb a hard binary? | `epsilon_tide ~ (M_p/m_bin)(a/R)^3` | Usually tiny for AU binaries at pc-kpc separation; process the carrier first |
| Is a carrier tidally limited? | `r_J ~ R[M_carrier/(3 M_enc)]^(1/3)` | Cluster disruption does not imply binary disruption |
| How fast does a cluster relax/segregate? | `t_rh ~ 0.138 N^(1/2) r_h^(3/2)/[sqrt(G mbar) ln Lambda]`; `t_seg ~ (mbar/m)t_rh` | Diagnostic against the detailed MC solver |
| Are stellar encounters relevant? | `Gamma_enc = n Sigma v`; `t_enc = 1/Gamma_enc` | Compare directly with `t_GW` |
| Is the binary hard? | `H = abs(E_b)/(mbar sigma^2)`, `E_b=-Gm1m2/(2a)` | `H >> 1` favors survival/hardening, not automatic destruction |
| Does an SMBH tidally threaten a delivered IMRI? | `r_H ~ R[m_bin/(3M_bh)]^(1/3)` | Require inner apoapsis well inside `r_H` |
| Which nuclear process wins? | compare `t_KL`, `t_GR`, `t_GW`, `t_enc` | Route to a calibrated single-SMBH or few-body kernel |

A cluster-merger model needs two separate quantities: a merger probability
conditioned on cluster/orbit/environment state, and a post-merger transition
kernel returning survival, exchange, ejection, prompt merger, or disruption
plus distributions of `a_final/a_initial`, `e_final`, and center-of-mass kick.
The two-IMBH case is a distinct few-body channel, not a density rescaling of
the one-IMBH case.

Immediately after a cluster merger, violent relaxation, anisotropy, rotation,
and unmixed subpopulations can violate equilibrium assumptions of a Henon-type
Monte Carlo solver. Use a short direct/calibrated transition, apply a settling
criterion such as virial balance plus stable half-mass radius, measure the
remnant state, and only then restart long-term MC evolution.

## 5. Local response library

Run the expensive local Monte Carlo on a controlled grid and persist a response
surface instead of recomputing it at every tree step. Candidate axes include:

```text
redshift
SMBH mass
NSC mass and structural parameters
AGN duty cycle/accretion state
disk migration scenario and gap condition
stellar/remnant population assumptions
```

For each cell persist:

- base/task seeds and repeat identity;
- intrinsic counts/rates with complete-repeat status;
- candidate masses, times, radii, and route/status distributions;
- conditional detector outputs and detector completeness;
- Monte Carlo uncertainty and interpolation support;
- model/version/provenance identifiers.

Pair scenarios through identical upstream draws. Interpolate only inside
validated support and return an explicit unavailable/out-of-support state
outside it.

## 6. Event-driven tree propagation

For each tree branch:

1. initialize or inherit source/carrier populations with immutable ancestry;
2. advance to the next discrete event: halo/galaxy merger, infall, stripping,
   disruption, AGN episode, compact-binary formation, recoil, or coalescence;
3. update the carrier and environment state;
4. query the local response library only for eligible AGN episodes;
5. schedule source evolution on its own clock;
6. terminate or transfer objects explicitly;
7. prevent descendants and surviving ancestors from being double counted.

The first controlling diagnostic is whether the environment has time to act:
compare source GW time, carrier dynamical-friction/stripping time, cluster
relaxation/encounter time, AGN episode duration, and remaining cosmic time.

## 7. Rate layers

Keep these outputs distinct:

```text
formation rate
intrinsic source-frame coalescence rate
observer-frame rate
detector-selected rate
```

A schematic observer-frame detector rate is

```text
dN_det/dt_obs
  = integral dz d(lambda)
      [dV_c/dz]
      R_src(lambda,z)/(1+z)
      p_det(lambda,z),
```

where the tree/host model supplies the weighted source rate and the detector
model supplies `p_det`. The factor `1/(1+z)` converts source time to observer
time. Do not treat the current per-host candidate rate or conditional LISA SNR
as this cosmological integral.

## 8. Model ladder

Add one scientifically interpretable uncertainty at a time:

| Stage | Added capability | Required comparison |
|---|---|---|
| M0 | local response, correct boundaries, paired seeds | current baseline |
| M1 | alternative gap/migration prescriptions | same disk and draws |
| M2 | remnant formation, retention, natal kicks | same host histories |
| M3 | improved waveform/detector selection | unchanged intrinsic sample |
| M4 | redshift-dependent occupation and host weighting | unchanged local kernel |
| M5 | merger-tree carrier histories | event/ancestry conservation |
| M6 | cluster, satellite, dual-SMBH, and other nuclear channels | explicit channel ledger |

The repository currently contains M0/M1-style local scenario machinery and a
first-order physical LISA detector model. It does not yet establish M4-M6.

## 9. Priority evidence

Before broadening the model, quantify:

- local Monte Carlo convergence and threshold-edge sensitivity;
- which physical stage dominates losses in each supported parameter bin;
- response interpolation error and unsupported regions;
- host occupation/duty weights and their provenance;
- carrier survival, disruption, and recoil bookkeeping;
- detector-model limitations relative to population uncertainty.

Do not add unsupported complexity to increase a rate. Each new layer needs a
separate status contract, conservation check, controlled comparison, and
persisted provenance.

## 10. Literature anchors

Use the literature as validation/calibration evidence, not as permission to
collapse distinct physical channels:

| Module | Starting references |
|---|---|
| EPS trees and statistics | [Parkinson, Cole & Helly 2008](https://doi.org/10.1111/j.1365-2966.2007.12517.x); [Jiang & van den Bosch 2014](https://academic.oup.com/mnras/article/440/1/193/1747580) |
| Satellite infall, decay, stripping | [Jiang et al. 2015](https://doi.org/10.1093/mnras/stv053); [Simha & Cole 2017](https://doi.org/10.1093/mnras/stx1942); [SatGen 2021](https://doi.org/10.1093/mnras/staa4034) |
| Cluster formation/evolution | [El-Badry et al. 2019](https://doi.org/10.1093/mnras/sty3007); [E-MOSAICS 2018](https://doi.org/10.1093/mnras/stx3124); [CMC catalog 2020](https://doi.org/10.3847/1538-4365/ab7919) |
| GW evolution and eccentric-lifetime caution | [Peters 1964](https://doi.org/10.1103/PhysRev.136.B1224); [Zwick et al. 2020](https://arxiv.org/abs/1911.06024) |
| Hard-binary scattering | [Quinlan 1996](https://doi.org/10.1016/S1384-1076(96)00003-6) |
| Nuclear delivery and processing | [Fragione & Leigh 2018](https://doi.org/10.1093/mnras/sty2233); [Arca-Sedda & Gualandris 2018](https://doi.org/10.1093/mnras/sty922) |
| Cluster mergers and massive BHs | [Souvaitzis, Rantala & Naab 2025](https://doi.org/10.1093/mnras/staf458); [Ishchenko et al. 2023](https://arxiv.org/abs/2308.12699); [Rantala et al. 2024](https://academic.oup.com/mnras/article/531/3/3770/7689219) |
| Cosmological compact-object rates | [Mapelli et al. 2017](https://doi.org/10.1093/mnras/stx2123); [Mapelli et al. 2022](https://academic.oup.com/mnras/article/511/4/5797/6533528) |
