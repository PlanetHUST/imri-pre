# Physical mechanisms and what this project has established

## 1. Aim and scope

The scientific aim is to construct a physically defensible high-redshift intermediate-mass-ratio inspiral (IMRI) source in an active galactic nucleus (AGN), then assess uncertainty in its formation yield and LISA detectability. The project currently follows **one specified host**, not a cosmological population.

The proposed chain is

$$
\text{NSC orbits}\longrightarrow\text{disk crossings}
\longrightarrow\text{gas capture/embedding}
\longrightarrow\text{compact-object migration}
\longrightarrow\text{GW-band entry}
\longrightarrow\text{LISA selection}.
$$

Each arrow is a physical bottleneck. More cluster objects do not automatically mean more compact objects, faster embedding does not guarantee inward migration, and more band entries do not guarantee stronger individual signals.

**Notation.** Let $M_\bullet$ be the central BH mass, $m$ the secondary mass, $q=m/M_\bullet$, $r$ the circular orbital radius, $a,e,i$ the pre-capture orbital elements, $H$ the disk scale height, $h=H/r$, $\Sigma$ the surface density, and $\rho_g$ the midplane gas density. Unless explicitly marked observer-frame, orbital times and rates are source-frame. Context equations below explain the physics; the linked source modules define the executable prescriptions.

## 2. Cluster population and access to the disk

For a single power-law spatial number density,

$$
n_\star(r)\propto r^{-\gamma_n},\qquad
\frac{dN}{dr}=4\pi r^2n_\star(r)\propto r^{2-\gamma_n}.
$$

A density law is not itself a radius probability density: the spherical-shell factor matters. Furthermore, a spatial-radius distribution is not generally a semi-major-axis distribution for eccentric orbits. The legacy sampler uses $p(a)\propto a^{2-\gamma_n}$ as its historical prescription. The opt-in Pan tracer instead uses a broken spatial profile and Eddington inversion in a Kepler potential; the two should not be identified.

A power-law stellar IMF obeys

$$
\frac{dN}{dm}=C m^{-\gamma_m},\qquad
\langle m\rangle=
\frac{\int_{m_{\min}}^{m_{\max}}m^{1-\gamma_m}\,dm}
     {\int_{m_{\min}}^{m_{\max}}m^{-\gamma_m}\,dm},\qquad
N\simeq\frac{M_{\rm NSC}}{\langle m\rangle}.
$$

At fixed total cluster mass, extending the IMF downward increases total object count while redistributing mass away from the high-mass tail. It need not increase the number of detectable BHs. A realistic BH inventory additionally requires a remnant map $m_{\rm BH}=g(m_{\rm birth},Z,\ldots)$, survival probabilities, ages and any pre-existing BH component. The archived fixed-mass stellar prescription does not provide that calibrated map.

An eccentric inclined orbit intersects the disk plane at two nodes with radii of the form

$$
r_{\pm}=\frac{a(1-e^2)}{1\pm e\cos\omega}.
$$

Only nodes inside the physical gas annulus contribute drag. Hence disk size, concentration, eccentricity and angular sampling jointly determine the initial supply. In the main benchmark, $R_{\rm out}=0.00323\,\mathrm{pc}$ whereas $R_{\rm NSC}=3.12\,\mathrm{pc}$: only about $0.62\%$ of initial objects are disk crossers. This is a geometry restriction, not an inverse migration timescale.

**Implemented:** explicit sampling, node selection, component budgets and optional population controls. **Not established:** a unique observationally calibrated z=10 cluster population or continuous relaxation-driven inflow. See [population prescriptions](published_populations.md) and the [formation-budget results](simulation_results_synthesis.md#2-formation-budget-where-are-objects-lost).

## 3. Disk structure couples capture and migration

Useful thin-disk relations are

$$
\Omega\simeq\sqrt{\frac{GM_\bullet}{r^3}},\qquad
H\sim\frac{c_s}{\Omega},\qquad
\nu=\alpha c_sH,\qquad
\dot M\simeq3\pi\nu\Sigma.
$$

The last relation is a steady-disk scaling away from boundary corrections, not a replacement for the project's pAGN structure solver. Temperature, opacity and pressure support set $c_s,H,\Sigma$ and thermal diffusion. Changing $\alpha$ or $\dot M$ therefore changes several capture and migration quantities simultaneously.

A useful self-gravity diagnostic is the Toomre parameter

$$
Q=\frac{c_s\kappa}{\pi G\Sigma},\qquad \kappa\simeq\Omega
$$

in a Keplerian disk. $Q\lesssim1$ signals susceptibility to local gravitational instability; extending a smooth disk beyond its allowed edge is not automatically a self-consistent model of that gas. The saved outer-radius experiment is a controlled boundary sensitivity, not a new fragmentation calculation.

**Implemented:** pAGN disk profiles, opacity-dependent quantities, and controlled parameter variations. **Still needed:** self-consistent disk families with stability and migration-validity diagnostics along each trajectory. A fixed absolute $\dot M$ scan in $M_\bullet$ is not a fixed Eddington-ratio scan, since $\dot M_{\rm Edd}\propto M_\bullet$ at fixed radiative efficiency.

## 4. Gas drag aligns and circularizes disk-crossing orbits

Crossing objects encounter gas with relative speed $v_{\rm rel}$. The numerical damping model uses an effective interaction radius and local drag time

$$
R_{\rm drag}=\max\!\left(R_\star,\frac{2Gm}{v_{\rm rel}^{2}}\right),\qquad
t_{\rm drag}=\frac{m}{\pi R_{\rm drag}^{2}\rho_gv_{\rm rel}}.
$$

For the BH branch the geometric stellar radius is zero; for stars the model uses $R_\star=R_\odot(m/M_\odot)^{0.8}$. The focusing radius above is the implemented crossing prescription, not a general subsonic Bondi/accretion formula. In particular it should not be extrapolated without validation to arbitrarily small relative speeds.

The force scaling is $F_{\rm drag}\sim\pi R_{\rm drag}^{2}\rho_gv_{\rm rel}^{2}$. Dissipation removes motion relative to the disk, changing inclination, eccentricity and orbital energy. Because gas acts during crossings rather than an entire orbit, a useful schematic orbit-averaged scaling is

$$
t_{\rm damp}^{-1}\sim\sum_{\text{active nodes}}
\frac{t_{\rm cross}}{P\,t_{\rm drag}},
$$

where $P$ is the orbital period. This explains why high density alone is insufficient: crossing duration, geometry and relative velocity matter.

The code numerically evolves crossing orbits and requires small inclination, $e\leq0.01$, radial containment and completion before AGN shutdown for a successful handoff. **Alignment alone is not embedding.** The budget audit found early stops after inclination damping with residual eccentricity; those failures require validated continued evolution, not an algebraic correction to the rate.

A second boundary concerns finite stars. Their approximate tidal disruption radius is

$$
r_t\sim R_\star\left(\frac{M_\bullet}{m}\right)^{1/3}.
$$

If disruption occurs before the hypothetical compact-object band entry, a point-mass stellar trajectory is not a BH IMRI. The current archived rates do not include a calibrated stellar disruption/consumption model.

**Source:** `src/physical_process/damp.py`. Hydrodynamically calibrated damping and its final-radius closure remain pending.

## 5. Migration: angular-momentum transport, gaps and traps

### 5.1 Default inward Type-I model

A circular low-mass secondary exchanges angular momentum with the disk through gravitational perturbations. In the default implementation,

$$
t_{\rm I}=\frac{M_\bullet^2h^2}{2m\Sigma r^2\Omega},\qquad
\dot r_{\rm gas}=-\frac{r}{t_{\rm I}}.
$$

At otherwise fixed disk properties, larger $m$ or $\Sigma$ gives faster inward drift, while a thicker disk slows it. This simple branch is always inward: it has no signed torque reversal, trap or Type-II switch. It is a controlled baseline, not proof of validity in a deep-gap regime.

### 5.2 Signed torques and thermal feedback

For $L=m\sqrt{GM_\bullet r}=mr^2\Omega$ at fixed masses,

$$
\dot r_{\rm gas}=\frac{2\Gamma}{mr\Omega},\qquad
\Gamma_{\rm I}=C_{\rm I}\Gamma_0,\qquad
\Gamma_0=\left(\frac{q}{h}\right)^2\Sigma r^4\Omega^2.
$$

The signed coefficient $C_{\rm I}$ depends on disk gradients and thermodynamic transport through Lindblad and corotation terms. A positive torque adds orbital angular momentum and drives outward motion. Thermal feedback changes the density perturbation around a luminous secondary and can oppose the inward gravitational torque.

For clarity, the thermal component actually used in the comparison can be written

$$
\Gamma_{\rm th}=W_{\rm valid}\Gamma_0\,
1.61\frac{\gamma_{\rm ad}-1}{\gamma_{\rm ad}}
\frac{x_c}{\lambda}\frac{1}{h}
\left[\frac{L}{L_c}\frac{4\mu}{1+4\mu}
-\frac{2\mu}{1+2\mu}\right]
\left(1-e^{-\sqrt{\alpha}\tau_{\rm eff}}\right),
$$

with

$$
\lambda=\sqrt{\frac{2\chi}{3\gamma_{\rm ad}\Omega}},\quad
x_c=-\frac{H^2}{3\gamma_{\rm ad}r}\frac{d\ln P}{d\ln r},\quad
L_c=\frac{4\pi Gm\rho_g\chi}{\gamma_{\rm ad}},\quad
\mu=\frac{\chi}{c_s\min(r_B,H)},\quad r_B=\frac{Gm}{c_s^2}.
$$

Here $\chi$ is thermal diffusivity, $\gamma_{\rm ad}$ the adiabatic index, and $L$ is set to the model's Eddington luminosity $4\pi Gmc/\kappa_{\rm es}$. The two terms in brackets represent saturated heating and cold contributions. The code checks length-scale, Hill-radius and optical-depth conditions through $W_{\rm valid}$; thermal extrapolation is off in the saved comparisons. Setting an invalid term to zero is an explicit model restriction, **not evidence that the actual physical thermal torque is zero**. Applying the same path to a star does not supply a calibrated stellar luminosity/feedback model.

### 5.3 Gaps and inertia-limited viscous transport

The continuous gap diagnostic uses

$$
K=\frac{q^2}{\alpha h^5},\qquad
f_{\rm gap}=\frac{1}{1+0.04K}.
$$

A deeper gap reduces gas available for linear gravitational torque. The implemented comparison combines terms as

$$
\Gamma=f_{\rm gap}\Gamma_{\rm I}+\Gamma_{\rm th}
+(1-f_{\rm gap})\Gamma_{\rm visc},
$$

where

$$
B=\frac{\Sigma r^2}{m},\qquad
\dot r_{\rm visc}=-\frac32\alpha h^2r\Omega\min(1,B),\qquad
\Gamma_{\rm visc}=\frac12mr\Omega\dot r_{\rm visc}.
$$

The factor $\min(1,B)$ slows transport when the secondary outweighs the characteristic local gas mass. This is a deliberately simple continuous prescription, not a calibrated sharp classical Type-II transition. The thermal term has its own optical-depth dependence: $\tau_{\rm eff}=f_{\rm gap}\tau$ in this branch. The ungapped control uses $\Gamma_{\rm I}+\Gamma_{\rm th}$ and $\tau_{\rm eff}=\tau$ instead.

### 5.4 Why a trap changes both yields and numerical difficulty

Let $v(r)=\dot r_{\rm gas}+\dot r_{\rm GW}$. A stable net-drift equilibrium satisfies

$$
v(r_*)=0,\qquad v'(r_*)<0,\qquad
\frac{d\delta r}{dt}\simeq v'(r_*)\delta r,\qquad
\tau_{\rm restore}=-\frac{1}{v'(r_*)}.
$$

Trapping can hold an object outside the GW band, possibly allowing stellar collapse before subsequent entry. A short restoring timescale makes the ODE stiff even if long-distance radial migration is slow. The project uses implicit Radau for the torque comparisons and retains DOP853 for analytical Type-I; this changes numerical integration, not the physical drift law. $\tau_{\rm restore}$ is not an NSC two-body relaxation time.

**Sources:** `src/physical_process/migration/{type_i,torque,gap}.py`. See [migration results](simulation_results_synthesis.md#3-migration-uncertainty-entry-numbers-and-delays-must-be-separated) for completed comparisons.

## 6. Transition from gas migration to gravitational radiation

For a circular extreme/intermediate mass-ratio orbit, the leading central-mass approximation is

$$
\dot r_{\rm GW}\simeq-\frac{64}{5}
\frac{G^3mM_\bullet^2}{c^5r^3},\qquad
\dot r=\dot r_{\rm gas}+\dot r_{\rm GW}.
$$

The general isolated two-body expression replaces $mM_\bullet^2$ by $mM_\bullet(M_\bullet+m)$. The project orbit engine uses a central-potential treatment; its detector chirp mass uses both component masses. Gas operates only during the prescribed AGN episode; a vacuum GW continuation can remain afterward. In the leading approximation,

$$
t_{\rm GW}(r_i\rightarrow r_f)
\simeq\frac{5c^5}{256G^3mM_\bullet^2}(r_i^4-r_f^4).
$$

The $r_i^4$ dependence explains how modest changes in release radius can create very long delays. Consequently, an initial host at z=10 can generate a much later band entry at a very different redshift. Gyr continuations with fixed masses and no calibrated host evolution are conditional trajectories, not established predictions for an evolving galaxy.

## 7. LISA selection: why formation and detection differ

The dominant circular GW frequency and cosmological mappings are

$$
f_{\rm src}\simeq\frac{1}{\pi}\sqrt{\frac{G(M_\bullet+m)}{r^3}},\qquad
f_{\rm obs}=\frac{f_{\rm src}}{1+z},\qquad
dt_{\rm obs}=(1+z)dt_{\rm src}.
$$

The detector uses a nonspinning endpoint

$$
f_{\rm ISCO,obs}=\frac{c^3}{6^{3/2}\pi G(M_\bullet+m)(1+z)}.
$$

Larger central mass or redshift pushes the endpoint downward. If it lies below the selected 1-mHz lower edge, this workflow cannot provide its usual band-entry classification; an unavailable event is not a measured zero rate. A different band/waveform study is a separate experiment.

For source chirp mass and its redshifted value,

$$
\mathcal M=\frac{(M_\bullet m)^{3/5}}{(M_\bullet+m)^{1/5}},\qquad
\mathcal M_z=(1+z)\mathcal M,
$$

the vacuum stationary-phase amplitude scales as

$$
|\widetilde h(f)|\propto
\frac{c}{D_L}\left(\frac{G\mathcal M_z}{c^3}\right)^{5/6}f^{-7/6},\qquad
\varrho^2=4\int_{f_{\rm low}}^{f_{\rm high}}
\frac{|\widetilde h(f)|^2}{S_n(f)}\,df.
$$

Here $\varrho$ is SNR (not gas density), $S_n$ includes the adopted LISA sensitivity/confusion model, and frequency limits reflect physical overlap and the actual mission window. A four-year mission does not imply four years of signal if the source crosses the usable band in months. For a fixed observed interval in the small-$q$ vacuum limit, $\varrho\propto m^{1/2}$ approximately; changing masses can also change the interval and residence time.

In the coupled model, residence time matters because stationary-phase power is proportional to $|h(t_f)|^2/|\dot f|$. Faster gas-driven passage can reduce SNR; it is not generally valid to insert a vacuum chirp throughout a gas-driven track. The project's replay/integration audit found this suppression negligible **for its measured in-band intervals**, not for all AGN inspirals.

The current selection is the hard cut $\varrho_{\rm averaged}\geq20$. A future orientation-marginalized probability would instead be

$$
p_{\rm det}(\theta)=\int d\omega\,p(\omega)
\,\mathbf 1[\varrho(\theta,\omega)\geq20],
$$

where $\theta$ denotes physical source parameters and $\omega$ orientation/sky variables. It is not equivalent to thresholding an averaged SNR. Spin and eccentric waveform extensions are likewise not silently included.

**Sources:** `src/lisa/{snr,detectability}.py`; [detector results and validation](simulation_results_synthesis.md#5-why-high-z-detections-vanish-in-the-reference).

## 8. Rates, replenishment and uncertainty

The saved normalization is

$$
\widehat R_X=\frac{N_X}{T_{\rm AGN}},\qquad
X\in\{\text{embedding, BH entry, detectable entry}\}.
$$

Embedding includes stars and BHs; BH band entry is not the same as ISCO consumption. Eligible consumption can trigger replacement from a finite donor pool, but this does not create an unlimited new reservoir or implement external gas/stellar inflow. Because delayed entries are retained, $N_{\rm entry}/T_{\rm AGN}$ is often best described as an episode yield divided by duty duration, not an instantaneous rate at the initial redshift.

For $n$ realizations,

$$
\overline R=\frac1n\sum_jR_j,\qquad
\mathrm{SEM}=\frac{s_R}{\sqrt n}.
$$

This measures finite-sampling scatter for fixed assumptions. It does not quantify population, disk, torque, remnant, waveform or numerical-model uncertainty. Zero detections with zero SEM are not a zero upper bound.

For comparison, a schematic cosmological calculation would require

$$
\frac{dN_{\rm det}}{dt_{\rm obs}}=
\int dz\,d\theta\,
\frac{dV_c}{dz}\frac{\mathcal R_{\rm entry}(z,\theta)}{1+z}
\,p_{\rm det}(z,\theta),
$$

where $\mathcal R_{\rm entry}$ must already include host abundance, episode history and delays per comoving volume and source time. This cosmological population integral has **not** been inferred from the present single-host runs.

## 9. Completed capabilities and implementation ownership

The project has implemented and exercised:

1. A reproducible single-host workflow with explicit seeds, per-object transitions, event flags, replenishment ledgers and saved metadata.
2. Numerical disk-crossing damping and coupled gas/GW evolution, including finite AGN lifetimes and delayed vacuum continuation.
3. Controlled host/redshift, IMF and disk/cluster parameter scans; optional published-population normalization and radial-law checks.
4. A four-scenario paired migration/refill comparison, with solver checks and identical primary damping records.
5. Event-level detector and population-budget audits, including independent SNR integration and end-to-end track replay.

The executable physics has these owners; scripts and visualization modules must not define competing physics:

| Stage | Source |
|---|---|
| Disk/opacity | `src/disk_quantity/pagn_disk_adapter.py`, `opacity.py` |
| Population and orbits | `src/physical_process/{population_models,sampling,nsc_orbital}.py` |
| Numerical capture | `src/physical_process/damp.py` |
| Default and signed migration | `src/physical_process/migration/` |
| Coupled orbit and band mapping | `src/physical_process/{orbital_evolution,gw_frequency_radius}.py` |
| Entry classification | `src/workflows/{event_rate,event_work}.py` |
| Finite-object refill | `src/workflows/{single_work,replenishment}.py` |
| Detector | `src/lisa/{snr,detectability}.py` |
| Configuration/schemas | `src/workflows/{config,outputs}.py` |

Optional SEVN, damping benchmarks and a separate programmatic Type-I/II comparison remain under `src/optional_models/`; none silently replaces the default physics. Eccentric post-kick sources outside the circular detector's applicability are unavailable, not zero-SNR sources.

Quantitative outcomes and their uncertainty are maintained only in [results and uncertainties](simulation_results_synthesis.md). Execution stages and saved-file definitions are in the [run guide](quickstart.md).
