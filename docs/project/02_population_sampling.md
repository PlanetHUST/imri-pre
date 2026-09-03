# Part 02 — Population Sampling

## 1. Idea

The NSC contains a stellar population whose compact remnants are the raw
material for IMRI candidates. Sampling produces, for one realization:

1. **stellar masses** and **formation times** (a proxy for stellar lifetime)
   consistent with the NSC total mass and metallicity;
2. **orbital elements** (`a, e, i, cos ω`) consistent with a power-law density
   profile around the SMBH;
3. a **geometric selection** of which orbits actually cross the AGN disk, using
   the true disk-plane node radii, not a semi-major-axis overlap test.

The main idea: *the NSC is the supply; the disk annulus is the filter. Only
orbits with at least one disk-plane node inside the physical annulus
`[R_in, R_out]` proceed to damping.* The resulting geometric fraction `f_geo` is
the first conditional efficiency of the pipeline.

## 2. Doing logic

### 2.1 Mass and lifetime sampling

Authoritative owner: `src/physical_process/sampling.py`.

`sample_efficient(mtot, Z)` picks the stellar population by metallicity:

- **Pop III** (`Z < 1e-4 Zsun`): top-heavy IMF over `13.6–197 Msun`, power-law
  index `1.77`; lifetimes from a cubic in `log10(mass)`.
- **Pop II** otherwise: Salpeter IMF over `0.01–250 Msun`, index `2.35`; stars
  below `1 Msun` are *discarded before summing mass* (a preserved modeling
  assumption); lifetimes from a separate cubic.

Batches are drawn until the retained mass reaches the NSC mass `M_nsc`. Masses
are in Msun; lifetimes are returned in **years** and converted to Myr by the
caller.

### 2.2 Orbital sampling

Authoritative owner: `src/physical_process/nsc_orbital.py::initial_star()`.

Given the SMBH mass, a minimum radius (explicitly passed; the workflow uses
`ORBITAL_SAMPLING_MIN_RADIUS_RG * r_g`, 6 rg), and the NSC radius as maximum:

- `a` is drawn from the inverse-CDF of the density profile `n(r) ∝ r^-γ`
  (`gamma = 7/4` by default);
- `e` from `sqrt(U)` (thermal);
- `i` uniform over `[10°, 170°]`;
- `cos ω` uniform in `[0, 1]`.

### 2.3 Disk-crossing selection

Authoritative owner: `src/physical_process/nsc_orbital.py`
(`orbital_node_radii_cm`, `orbital_nodes_in_disk_annulus`, `disk_crossing_mask`)
called from `src/workflows/population.py::sample_disk_crossing_population()`.

The two disk-plane node radii of an orbit are:

```
r_node,1 = a(1-e²) / (1 - e cos ω)
r_node,2 = a(1-e²) / (1 + e cos ω)
```

An orbit **crosses the disk** iff at least one node lies in the physical
annulus:

```
crosses = (R_in ≤ r_node,1 ≤ R_out)  OR  (R_in ≤ r_node,2 ≤ R_out)
```

with `R_in = disk_inner_radius_cm`, `R_out = outer_radius_cm` (the Q=1
boundary). `use_disk_inner_radius=False` keeps only the outer-edge test.

Note that this is **not** an interval-overlap test (`min < R_out and max >
R_in`): an orbit with nodes at `5` and `200` does **not** cross `[10, 100]`
even though `5 < 100` and `200 > 10`.

### 2.4 Result payload

`PopulationSelection` carries `initial` and `disk_crossing` populations plus
`mean_relaxation_time_myr` (mean of `nsc_orbital.relaxaion_timescale_s` over
the selected orbits, /Myr). `f_geo = disk_crossing_count / total_count`.

## 3. Key parameters (config-owned)

| Parameter | Default | Meaning |
|---|---|---|
| `metallicity` | 1e-5 | selects Pop III vs Pop II |
| `initial_stellar_density_gamma` | 7/4 | orbital density slope |
| `nsc_radius_*` | 3.31 pc / 3.6e6 / 0.321 | NSC mass–radius relation |
| `ORBITAL_SAMPLING_MIN_RADIUS_RG` | 6.0 | lower bound of `a` sampling |

## 4. Caveats

- **Sampled masses are initial stellar masses, not remnants.** The current
  worker begins objects as `Star` and preserves the sampled initial mass; there
  is no stellar-to-remnant population model yet. Candidate mass distributions
  must not be read as a final BH mass function
  ([`../reference/research_roadmap.md`](../reference/research_roadmap.md)).
- **The disk-crossing predicate is node-based, not semi-major-axis based.** An
  eccentric orbit can cross the disk while its `a` is far outside the Q=1 edge;
  a low-`a` orbit can fail to cross. This distinction is a governing decision
  from [`../reference/physical_model.md`](../reference/physical_model.md).
- **Selection uses the physical annulus.** `R_out` is the Q=1 boundary from the
  environment part, never the extrema of padded solver arrays.
- **Determinism is downstream of sampling order.** Paired runs must keep the
  random-number call order fixed; changing the order breaks seeded parity even
  with the same `task_seed`.
- **Damping (part 03) decides real embedding.** Geometric crossing here is only
  eligibility; many selected orbits will not finish as embedded objects.
