# IMRI consumption-driven replenishment

## Historical basis

Commit `0f1c751` kept a TDE `dyn_resupply` path that scheduled copies from
two-body relaxation times, while its IMRI demo used only the initial cohort.
Commit `2067587` made the IMRI workflow primary and removed that TDE copy
path. The current IMRI relaxation-time value is a population diagnostic. This
plan restores a supply stage for IMRI without reviving relaxation-clock copy
scheduling or TDE event physics.

## Physical contract

One sampled disk-crossing orbit is a supply channel. Its initial occupant is
drawn at the start of the AGN episode. A channel reopens only when an eligible
compact object reaches the SMBH ISCO during that episode. LISA band entry is
recorded earlier and does not consume the channel.

Each reopening draws one distinct object from the initially sampled NSC objects
that were not selected as primary disk crossers. The donor adopts the vacant
channel's initial orbit and starts damping at the preceding ISCO time. A donor
can be used once. No external inflow, relaxation-clock injection, or arbitrary
copy cap is included. The existing stellar lifetime is absolute since episode
start, so a donor may already be a compact remnant when injected.

The current two-body relaxation calculation remains an environmental
diagnostic. The AGN duty cycle bounds new injections; an IMRI can enter the
LISA band after the gas episode, but that later event cannot open a new channel
in this episode. The same damping, coupled gas/GW orbit, event classification,
and LISA detector code evaluate primary and replacement objects.

## Implementation parts

1. Record the first ISCO time by continuing the existing coupled orbit from
   band entry. Preserve a distinct status when ISCO is not reached or its
   continuation fails; a failed continuation makes the population rate
   incomplete because a replacement might be missing.
2. Schedule channels in consumption-time order. Draw donors without replacement
   from the finite NSC reservoir; keep a source-count invariant and a causal
   consumption/injection ledger.
3. Run each injected donor through the production damping worker and IMRI event
   evaluator. Persist primary/replacement identity, times, per-repeat source
   counts, the ledger, and the model choice in metadata.
4. Check a controlled causal chain, exhausted supply, events after the duty
   horizon, output schema, and a reduced seeded single-halo run.

## Interpretation

This is a first-order phase-space channel model. Reusing an orbital template
approximates scattering into an empty channel; it does not solve angular
momentum diffusion. The initial sampled NSC is the entire finite donor pool.
Its size, IMF, and mass approximation retain their present scientific limits.

## Implementation status

The scheduler, coupled-orbit ISCO continuation, configuration switch, event
identity, finite-source ledger, summary columns and documentation are in the
working tree. Static Python parsing in `sevnpy_env` and `git diff --check`
completed. Numerical and end-to-end tests have not been run in this change.
