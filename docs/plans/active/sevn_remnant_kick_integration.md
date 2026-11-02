# Single-halo remnant and natal-kick integration

## Change contract

The control is the current fixed-mass single-halo workflow. The changed axis is
stellar evolution: SEVN supplies collapse age, remnant type, gravitational
remnant mass, and one natal-kick draw. IMRI owns the Cartesian kick-to-osculating-orbit transformation in
`physical_process.natal_kick`, migrated from `adv_damp`. IMRI continues to own NSC sampling,
the solved pAGN disk, orbital damping, gas/GW migration, LISA classification,
and finite consumption-driven donor supply. A paired run uses the same host,
disk, selected source IDs, donor RNG policy, and task seeds. Donor chains can
diverge after the first change in ISCO consumption.

The production default remains the fixed-mass control. SEVN is opt-in and its
table paths and kick policy must be recorded. IMRI metallicity is in solar
units; SEVN receives absolute `Z = metallicity_zsun * 0.0145`. All time stamps
are absolute Myr since the AGN episode began. Remnant mass is gravitational
mass in solar masses. SEVN kick components are km/s and are converted once to
cm/s for orbital evolution. An isolated SEVN star has arbitrary kick axes;
the drawn speed is retained and its direction is mapped isotropically to the
orbital frame using an object-local RNG.

The installed SEVN backend represents pair-instability disruption as remnant
type `-1` with a nonfinite final `Mass`; its SN log gives zero remnant mass.
The adapter treats this as a physical massless outcome.

## Baseline and dependency report

Read existing `imri_averaged_results.csv`, `imri_repeat_results.csv`,
`imri_events.csv`, `imri_replenishment_ledger.csv`, and metadata. Report the
input contract, denominator, stage counts and rates by repeat, SNR-status
counts, and retrospective SNR cuts at 15 and 20. Never infer an unavailable
SNR as zero. The representative control is the saved 20-repeat
`torque_z15_duty10` run with `M_SMBH=1e5 Msun`, `M_NSC=2e5 Msun`,
`Z=1e-5 Zsun`, and a 10 Myr AGN episode.

## Part 1: external stellar endpoint

Load SEVN once per worker process from explicit hydrogen and helium table
paths. Construct each star from its sampled ZAMS mass, absolute metallicity,
and a stable source-ID seed. Read the absolute collapse age, remnant type,
gravitational mass, and effective kick from the same SEVN evolution. Do not
apply an independent Fryer remnant or Maxwellian kick draw. Record
the SEVN model options and table identity. No-remnant and non-BH endpoints are
separate physical outcomes. A donor injected after its collapse starts as a
remnant at its injection time; its historical kick is not re-applied to the
assumed channel-template orbit.

## Part 2: orbital event

During damping, stop exactly at collapse, change mass/type, apply the kick
through the local orbital transform, recheck boundness and embedding, then resume BH damping
when necessary. An embedded star continues IMRI's migration until collapse;
it must not wait at a fixed radius. Split circular migration at the collapse
time, apply the mass and velocity jump, then continue the same gas/GW model if
the remnant is still eligible for circular migration. Otherwise return to
damping while the AGN is active. Eccentric post-kick vacuum inspirals need an
explicit unavailable status until an eccentric detector model is connected.
For a kicked BH that does not re-embed, evolve both semi-major axis and
eccentricity with the Peters invariant to a generous screening frequency:
four times the circular GW frequency evaluated at pericenter, at `z=0`.
If this proxy still needs longer than the remaining cosmic time, classify
no band entry under that explicit convention. A potentially entering
eccentric orbit stays unavailable until an eccentric waveform is connected.

## Part 3: outputs and checks

Persist ZAMS and remnant masses, collapse time and phase, kick speed, post-kick
elements, and a mutually exclusive stellar outcome. Keep the original event
and rate columns meaningful. Compare fixed-mass, SEVN/no-kick, and SEVN/kick
runs on paired source IDs. Check mass and time units, repeatable object-local
kicks, bound/unbound orbital limits, donor uniqueness, count conservation,
status semantics, and a reduced seeded end-to-end run. Any unsupported branch
must make the relevant rate incomplete rather than silently remove a source.

## Implemented interface and remaining model boundary

The opt-in controls are `--stellar-evolution-model sevn`,
`--natal-kick-model sevn|none`, and explicit H/He table paths. The fixed-mass
default remains the control. Each selected source and replacement donor has
one `imri_stellar_ledger.csv` row. Metadata records SEVN supernova options,
package versions, and table hashes. `scripts/report_stellar_integration.py`
compares saved runs without executing physics.

Pre-collapse drag still uses sampled ZAMS mass, while the ledger also records
SEVN's pre-collapse mass and phase. A donor injected after its collapse enters
the assumed channel-template orbit as a remnant; its historical kick is not
reapplied to that orbit. The LISA model remains circular and nonspinning. A
bound eccentric kicked object that can plausibly enter the band would make
the rate incomplete until an eccentric orbit/detector treatment is connected.

## Verification completed

The fixed-mass default reproduced the saved control's repeat-0 event rows,
source IDs, and every shared event field exactly. The two SEVN modes retained
all 1,337 paired primary source IDs and identical stellar endpoints and seeds.
The full test suite, split-orbit and collapse-stop checks, the independent
Peters ODE check, artifact status checks, and `git diff --check` pass. The
20-repeat comparison and its dependency tables are saved in
`output/sevn_remnant_kick_20261027/report.md`; the concise scientific result
is in `docs/reports/sevn_remnant_kick_baseline_20261027.md`.
