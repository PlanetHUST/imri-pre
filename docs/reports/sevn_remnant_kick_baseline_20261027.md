# First single-halo SEVN remnant and kick comparison

This is a conditional rate for **one modeled halo and one 10 Myr AGN episode**,
not a cosmic rate. The controlled host is `M_SMBH = 1e5 Msun`,
`M_NSC = 2e5 Msun`, `z = 15`, `Z = 1e-5 Zsun`, with the signed torque model,
20 paired repeat seeds from base seed `12345`, the current finite donor policy,
and the same disk-crossing population in all three modes. The fixed-mass
control is the saved 2026-10-23 replenishment run. Both SEVN modes use the
local Pop III H tables and PARSEC He tables. The kick mode applies SEVN's
effective kick speed with `adv_damp`'s orbital transform. SNR uses the existing
four-year LISA window and circular waveform.

| Mean per episode | Fixed-mass control | SEVN, no kick | SEVN, kick |
| --- | ---: | ---: | ---: |
| Sampled NSC sources | 4,521 | 4,521 | 4,521 |
| Disk-crossing sources | 66.85 | 66.85 | 66.85 |
| Embedded stars | 12.5 | 11.3 | 11.3 |
| Embedded BHs | 5.10 | 3.25 | 3.00 |
| BHs formed during migration | 0.75 | 1.15 | 0.80 |
| Intrinsic IMRIs | 5.10 | 2.65 | 2.70 |
| Intrinsic rate [halo Myr]^-1 | 0.510 | 0.265 | 0.270 |
| Detections at SNR 20 | 0.10 | 0 | 0 |
| Donors injected after ISCO consumption | 4.90 | 2.55 | 2.60 |

All 20 repeat-level intrinsic and detector rate statuses are `complete` in
each mode. The SEVN no-kick minus fixed-mass intrinsic count is
`-2.45 ± 0.35` per episode (paired mean ± Monte Carlo standard error).
The kick minus no-kick difference is `+0.05 ± 0.11` per episode. These errors
describe repeat-to-repeat sampling only, not uncertainty in the stellar,
disk, or detector prescriptions. The 20 kick repeats apply 128 nonzero kicks;
45 produce unbound objects. Among the bound non-embedded cases, a Peters
`a,e` evolution to a generous pericenter-frequency proxy (four times twice
the orbital frequency at pericenter) excludes band entry by `z=0` for the 78 cases
so classified. The shortest proxy time is about 45,874 Myr, versus at most
13,527 Myr remaining to `z=0`. An eccentric object with plausible entry
would make the rate incomplete instead; this proxy is not a waveform/SNR
calculation.

The old run has two detections across 20 episodes; both SEVN modes have zero.
Zero in 20 leaves a one-sided 95% Poisson upper limit of about 0.15 detections
per episode, so the detector-rate difference is not well measured. Retrospective
cuts at SNR 15 and 20 both give `2, 0, 0` total detections across the three
modes. The SNR cut does not determine the much larger intrinsic-rate change.

The [full generated report](../../../output/sevn_remnant_kick_20261027/report.md)
shows stage counts, paired primary-source membership, status partitions, table
hashes, and package versions. Regenerate it from saved artifacts with
`scripts/report_stellar_integration.py`; this script does not advance the
simulation or its RNG. The [integration plan](../plans/active/sevn_remnant_kick_integration.md)
states the event, unit, and remaining-model contracts.
