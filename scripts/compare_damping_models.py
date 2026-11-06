#!/usr/bin/env python3
"""Reproducible first-stage damping benchmark, independent of event-rate code.

Run with PYTHONPATH=src python scripts/compare_damping_models.py --help.
No migration, stellar/remnant resampling, mergers, detector, or rate fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from physical_process import constants as c
from physical_process.damping_models import (
    DampingInput, RepoDampingAdapter, XueDampingAdapter, OutcomeStatus,
    REPO_MODEL, XUE_MODEL, SCATTERING, orbital_invariants,
)
from physical_process.damping_benchmark import (
    reference_source, resolve_fixture, adapter_for_profile, source_on_profile,
)
from physical_process.damping_statistics import summarize_flux

SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def csv_rows(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text('status\nno_rows\n')
        return
    # Cache JSON sorts object keys; use the same schema ordering on fresh and
    # resumed paths so CSV bytes do not depend on dictionary insertion order.
    columns = sorted({key for row in rows for key in row})
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list, tuple)) else v
                             for k,v in row.items()})
    temp.replace(path)


def code_hash():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'scripts/damping_comparison_figures.py',
             *(root/'src/physical_process').glob('damp*.py'),
             root/'src/physical_process/constants.py', root/'src/physical_process/agn.py',
             root/'src/disk_quantity/pagn_disk_adapter.py']
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class Evaluator:
    """Atomic per-source checkpoint, keyed by full input + code + disk digest."""
    def __init__(self, output, resume, identity, max_iter):
        self.cache = output/'checkpoints'
        self.cache.mkdir(exist_ok=True)
        self.resume = resume
        self.identity = identity
        self.max_iter = max_iter
        self.reused = 0
        self.evaluated = 0

    def pair(self, x, *, experiment, profile=None, history=False):
        results, histories = [], []
        for model in (RepoDampingAdapter(adapter_for_profile(x, profile)), XueDampingAdapter()):
            profile_hash = None if profile is None else digest(dict(
                arrays={name: getattr(profile, name).tolist() for name in
                    ('radius_cm','surface_density_g_cm2','aspect_ratio','density_g_cm3','sound_speed_cm_s')},
                inner_radius_cm=profile.inner_radius_cm,outer_radius_cm=profile.outer_radius_cm))
            identity = dict(input=asdict(x), model=model.model_id, experiment=experiment,
                            run_identity=self.identity, disk_profile_sha256=profile_hash,
                            history=history, max_iter=self.max_iter)
            key = digest(identity)
            path = self.cache/(key+'.json')
            if self.resume and path.exists():
                saved = json.loads(path.read_text())
                if saved['identity'] != identity:
                    raise ValueError(f'checkpoint identity mismatch: {path}')
                row, tracks = saved['row'], saved['history']
                self.reused += 1
            else:
                outcome = model.evolve(x, max_iter=self.max_iter, keep_history=history)
                row = asdict(x)
                row.update(experiment=experiment, model_id=outcome.model_id,
                    calibration_id=outcome.calibration_id, configuration=outcome.configuration,
                    status=outcome.status.value, native_status=outcome.native_status,
                    t_embed_myr=outcome.t_embed_myr, r_embed_cm=outcome.r_embed_cm,
                    inclination_at_embed_rad=outcome.inclination_at_embed_rad,
                    eccentricity_at_embed=outcome.eccentricity_at_embed,
                    delta_energy_erg=outcome.delta_energy,
                    delta_angular_momentum_g_cm2_s=outcome.delta_angular_momentum,
                    validity_flags=list(outcome.validity_flags),
                    completeness_flags=list(outcome.completeness_flags),
                    validity_coverage=model.validity_mask(x).coverage,
                    censor_time_myr=outcome.censor_time_myr,
                    final_time_myr=outcome.final_state.time_s/c.Myr,
                    final_radius_cm=outcome.final_state.semi_major_axis_cm,
                    final_eccentricity=outcome.final_state.eccentricity,
                    final_inclination_rad=outcome.final_state.inclination_rad,
                    effective_scattering_di_dt_rad_s=outcome.effective_scattering_di_dt_rad_s,
                    effective_scattering_delta_energy=outcome.effective_scattering_delta_energy,
                    effective_scattering_delta_angular_momentum=outcome.effective_scattering_delta_angular_momentum,
                    input_sha256=digest(asdict(x)), checkpoint_key=key)
                local = model.inclination_damping_rate(x)
                row.update(initial_di_dt_rad_s=local.value, local_rate_status=local.status,
                           effective_scattering_status='not_included_no_calibrated_term')
                tracks = []
                for state in outcome.history:
                    E,L = orbital_invariants(state,x)
                    tracks.append(dict(experiment=experiment, source_id=x.source_id,
                        model_id=model.model_id, time_myr=state.time_s/c.Myr,
                        radius_cm=state.semi_major_axis_cm, eccentricity=state.eccentricity,
                        inclination_rad=state.inclination_rad,
                        energy_erg=E if model.model_id==REPO_MODEL else None,
                        angular_momentum_g_cm2_s=L if model.model_id==REPO_MODEL else None,
                        radius_status='integrated' if model.model_id==REPO_MODEL else 'fixed_reference_assumption',
                        validity_flags=list(outcome.validity_flags)))
                write_json(path, dict(identity=identity, row=row, history=tracks))
                self.evaluated += 1
            results.append(row)
            histories.extend(tracks)
        assert results[0]['input_sha256']==results[1]['input_sha256']
        return results, histories


def build_cohort(args):
    if args.input_json:
        data = json.loads(args.input_json.read_text())
        sources = [DampingInput(**row) for row in data]
        if not sources or any(x.invalid_flags() for x in sources):
            raise ValueError('input JSON must be a nonempty list of valid DampingInput records')
        keys = [(x.realization,x.source_id) for x in sources]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate source IDs within realization')
        return sources
    sources = []
    for rep in range(args.repeats):
        seed = int(np.random.SeedSequence([args.seed, rep]).generate_state(1,dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        inc = np.deg2rad(rng.uniform(2.,15.,args.sources))
        # Overlap with Whitehead Fig5 low-i mtilde span: Sigma<=Sigma0
        # gives mtilde<=.1705<.199 throughout the embedding path.
        sigma = 1.6e4*np.exp(rng.uniform(np.log(.25),0.,args.sources))
        weights = rng.uniform(.5,1.5,args.sources)
        # Fix total weighted cohort size per realization; this is a declared
        # synthetic fixture normalization, shared exactly by both models.
        weights *= args.sources/weights.sum()
        for j in range(args.sources):
            sources.append(reference_source(inclination_rad=float(inc[j]),
                surface_density_g_cm2=float(sigma[j]), source_weight=float(weights[j]),
                disk_lifetime_s=args.lifetime_myr*c.Myr, seed=seed, realization=rep,
                source_id=f'r{rep:03d}_s{j:05d}'))
    return sources


def parameter_cases():
    x = reference_source()
    specs = {
        'inclination_rad': np.deg2rad([.2, .5, 1, 2, 4, 8, 12, 15, 20, 60, 120]),
        'radius_cm': x.radius_cm*np.geomspace(.25,4,7),
        'eccentricity': [0.,.001,.01,.1,.5,.9],
        'compact_object_mass_g': c.msun*np.array([5,10,20,25,30,50,100]),
        'smbh_mass_g': c.msun*np.array([1e5,1e6,4e6,1e7,1e8]),
        'surface_density_g_cm2': 1.6e4*np.array([.125,.25,.5,1,2,4]),
        'aspect_ratio': x.aspect_ratio*np.array([.25,.5,1,2,4]),
        'viscosity_alpha': [.01,.03,.1,.3],
        'disk_lifetime_s': c.Myr*np.geomspace(1.e-6,1.,9),
    }
    for parameter, values in specs.items():
        for k, value in enumerate(values):
            if parameter=='aspect_ratio':
                y=replace(x, scale_height_cm=float(value*x.radius_cm))
            elif parameter=='radius_cm':
                y=replace(x, radius_cm=float(value),scale_height_cm=float(value*x.aspect_ratio))
            else:
                y=replace(x, **{parameter: float(value)})
            yield parameter,float(value),resolve_fixture(replace(y,source_id=f'scan_{parameter}_{k}'))


def sensitivity_rows(evaluate, profile=None):
    rows=[]
    x=reference_source()
    eps=.03
    for parameter in ('inclination_rad','radius_cm','compact_object_mass_g','smbh_mass_g',
                      'surface_density_g_cm2','aspect_ratio','viscosity_alpha','disk_lifetime_s'):
        endpoints=[]
        for sign in (-1,1):
            factor=np.exp(sign*eps)
            if parameter=='aspect_ratio':
                y=replace(x,scale_height_cm=x.scale_height_cm*factor)
            elif parameter=='radius_cm':
                y=replace(x,radius_cm=x.radius_cm*factor,scale_height_cm=x.scale_height_cm*factor)
            else:
                y=replace(x,**{parameter:getattr(x,parameter)*factor})
            result,_=evaluate.pair(resolve_fixture(replace(y,source_id=f'sensitivity_{parameter}_{sign}')),
                                   experiment='local_log_sensitivity')
            endpoints.append(result)
        for m in range(2):
            a,b=endpoints[0][m],endpoints[1][m]
            good=all(z['status']==OutcomeStatus.EMBEDDED and z['t_embed_myr']>0 for z in (a,b))
            rates=[abs(z['initial_di_dt_rad_s']) if z['initial_di_dt_rad_s'] is not None else None for z in (a,b)]
            rows.append(dict(parameter=parameter,model_id=a['model_id'],
                dln_tembed_dln_parameter=(np.log(b['t_embed_myr']/a['t_embed_myr'])/(2*eps) if good else None),
                dln_abs_rate_dln_parameter=(np.log(rates[1]/rates[0])/(2*eps)
                    if all(v is not None and v>0 for v in rates) else None),
                status='evaluated' if good else 'unavailable_endpoint',
                minus_status=a['status'],plus_status=b['status'],
                step_log=eps,conditions='one_at_a_time_local_fixture; alpha has no explicit drag term'))
    # e=0 is a boundary, so logarithmic eccentricity derivative is undefined.
    rows.extend(dict(parameter='eccentricity',model_id=m,status='undefined_log_at_e_zero',
                     dln_tembed_dln_parameter=None) for m in (REPO_MODEL,XUE_MODEL))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/damping_benchmark'))
    parser.add_argument('--sources',type=int,default=64,help='sources per synthetic realization')
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--seed',type=int,default=260529305)
    parser.add_argument('--lifetime-myr',type=float,default=.1)
    parser.add_argument('--bootstrap-draws',type=int,default=2000)
    parser.add_argument('--max-iter',type=int,default=50000)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--input-json',type=Path,help='exact serialized DampingInput list; no resampling')
    parser.add_argument('--skip-pagn-audit',action='store_true')
    args=parser.parse_args()
    if min(args.sources,args.repeats,args.bootstrap_draws,args.max_iter)<=0 or args.lifetime_myr<=0 or args.seed<0:
        parser.error('positive counts/lifetime and nonnegative seed required')
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    sources=build_cohort(args)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('resume','output_dir')}
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pagn')}
    identity=dict(schema_version=SCHEMA_VERSION,code_sha256=code_hash(),settings=settings,
                  runtime_versions=versions, python=platform.python_version(),
                  cohort_sha256=digest([asdict(x) for x in sources]))
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
        raise ValueError('output directory belongs to a different experiment/code; choose a new directory')
    metadata=dict(identity=identity,python=platform.python_version(),executable=sys.executable,
        versions=versions,command=' '.join(sys.argv),seed_schedule='SeedSequence([base_seed, realization]) -> uint64 -> default_rng',
        source_population='exact_input_json' if args.input_json else 'controlled circular compact-object combined-fit fixture; not NSC IMF or event-rate population',
        normalization='weights sum to --sources per realization for generated fixture',
        stellar_remnant_mapping='not invoked; identical compact-object masses supplied to both models',
        disk='local state fixture plus separate existing pAGN Sirko audit',
        assumptions=['Xue radius fixed reference, no independently calibrated total E/L',
                     'repo density fixture Sigma/(2H); production audit uses solved rho',
                     'both use threshold i<=H/R; repo also e<=.01 and containment',
                     'no effective stellar encounters, no relaxation dynamics'],
        calibration_sources=['https://arxiv.org/abs/2605.29305','https://arxiv.org/abs/2505.23899','https://arxiv.org/abs/2505.23739'])
    write_json(manifest,metadata)
    write_json(output/'matched_inputs.json',[asdict(x) for x in sources])
    evaluate=Evaluator(output,args.resume,identity,args.max_iter)
    population=[]
    for k,x in enumerate(sources):
        pair,_=evaluate.pair(x,experiment='matched_population')
        population.extend(pair)
        if (k+1)%64==0:
            print(f'population: {k+1}/{len(sources)} matched sources',flush=True)
    csv_rows(output/'damping_source_outcomes.csv',population)
    lifetime=max(max(x.disk_lifetime_s for x in sources)/c.Myr,1.e-12)
    edges=np.r_[0.,np.geomspace(min(lifetime/1.e5,1.e-7),lifetime,21)]
    # Unequal imported lifetimes remain explicitly incomplete beyond censoring.
    flux=summarize_flux(population,edges,bootstrap_seed=args.seed,bootstrap_draws=args.bootstrap_draws)
    csv_rows(output/'damping_flux_comparison.csv',flux)
    tracks=[]; fixed=[]
    for inc in (1.,2.,5.,8.,12.,15.):
        x=reference_source(inclination_rad=float(np.deg2rad(inc)),source_id=f'trajectory_{inc:g}deg')
        pair,history=evaluate.pair(x,experiment='fixed_history',history=True)
        fixed.extend(pair);tracks.extend(history)
    csv_rows(output/'damping_fixed_sources.csv',fixed)
    csv_rows(output/'damping_trajectories.csv',tracks)
    scan=[]
    for parameter,value,x in parameter_cases():
        pair,_=evaluate.pair(x,experiment='parameter_scan')
        for row in pair:
            row.update(scan_parameter=parameter,scan_value=value)
        scan.extend(pair)
    csv_rows(output/'damping_parameter_scan.csv',scan)
    sensitivities=sensitivity_rows(evaluate)
    csv_rows(output/'damping_sensitivities.csv',sensitivities)
    # Lifetime stress tests relative to each model's uncensored reference time.
    base=reference_source(disk_lifetime_s=100*c.Myr,source_id='boundary_reference')
    pair,_=evaluate.pair(base,experiment='lifetime_reference')
    boundaries=[]
    for target in pair:
        if target['t_embed_myr'] is None:
            continue
        for factor in (.001,.1,.999,1.,1.001,10.,100.):
            x=replace(base,disk_lifetime_s=target['t_embed_myr']*factor*c.Myr,
                      source_id=f"boundary_{target['model_id']}_{factor}")
            results,_=evaluate.pair(x,experiment='lifetime_boundary')
            for row in results:
                row.update(lifetime_reference_model=target['model_id'], lifetime_factor=factor)
            boundaries.extend(results)
    csv_rows(output/'damping_lifetime_boundaries.csv',boundaries)
    # Scattering request is documented as unavailable for each adapter.
    unsupported=[]
    for model in (RepoDampingAdapter(adapter_for_profile(base),SCATTERING),XueDampingAdapter(SCATTERING)):
        o=model.evolve(base)
        unsupported.append(dict(model_id=model.model_id,configuration=SCATTERING,
            status=o.status.value,validity_flags=o.validity_flags,
            di_dt_rad_s=o.effective_scattering_di_dt_rad_s,delta_energy=o.effective_scattering_delta_energy,
            delta_angular_momentum=o.effective_scattering_delta_angular_momentum))
    csv_rows(output/'damping_scattering_availability.csv',unsupported)
    audit=[];disk_record={'status':'skipped_by_cli'}
    if not args.skip_pagn_audit:
        from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
        try:
            profile=build_pagn_disk_profile(1.e7,.1,n_resolution=1000,model='sirko')
            disk_record=dict(status='evaluated',model='existing_pagn_sirko',alpha=.1,eddington_ratio=.1,
                n_resolution=1000,inner_radius_cm=profile.inner_radius_cm,outer_radius_cm=profile.outer_radius_cm)
            write_json(output/'pagn_disk_profile.json',dict(radius_cm=profile.radius_cm.tolist(),
                surface_density_g_cm2=profile.surface_density_g_cm2.tolist(),aspect_ratio=profile.aspect_ratio.tolist(),
                density_g_cm3=profile.density_g_cm3.tolist(),sound_speed_cm_s=profile.sound_speed_cm_s.tolist(),
                provenance=disk_record))
            for rfrac in (.2,.5,.8):
                for eccentricity in (0.,.3,.7):
                    x=source_on_profile(reference_source(radius_cm=profile.outer_radius_cm*rfrac,
                        eccentricity=eccentricity,inclination_rad=np.deg2rad(8.),
                        source_id=f'pagn_{rfrac}_{eccentricity}',disk_lifetime_s=args.lifetime_myr*c.Myr),profile)
                    results,history=evaluate.pair(x,experiment='existing_pagn_domain_audit',profile=profile,history=True)
                    audit.extend(results);tracks.extend(history)
            csv_rows(output/'damping_pagn_audit.csv',audit)
            csv_rows(output/'damping_trajectories.csv',tracks)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            disk_record={'status':'unavailable','error':f'{type(exc).__name__}: {exc}'}
    write_json(output/'pagn_audit_status.json',disk_record)
    # Weighted radius-time joint distribution. All non-events remain in source CSV.
    joint=[]
    # Include every actual event radius, including externally supplied cohorts.
    event_radii=[r['r_embed_cm'] for r in population if r['r_embed_cm'] is not None]
    radial_edges=base.radius_cm*np.array([0.,.9,.95,.975,.99,1.000001,1.1,10.])
    if event_radii:
        radial_edges=np.unique(np.r_[radial_edges, min(event_radii)*.999, max(event_radii)*1.001])
    for model in (REPO_MODEL,XUE_MODEL):
        for rep in sorted({x.realization for x in sources}):
            events=[r for r in population if r['model_id']==model and r['realization']==rep and r['status']==OutcomeStatus.EMBEDDED]
            hist,_,_=np.histogram2d([r['t_embed_myr'] for r in events],[r['r_embed_cm'] for r in events],
                bins=(edges,radial_edges),weights=[r['source_weight'] for r in events])
            for ti in range(len(edges)-1):
                for ri in range(len(radial_edges)-1):
                    joint.append(dict(model_id=model,realization=rep,time_left_myr=edges[ti],time_right_myr=edges[ti+1],
                        radius_left_cm=radial_edges[ri],radius_right_cm=radial_edges[ri+1],weight=hist[ti,ri],
                        interpretation='observed_events_only; see source statuses for completeness'))
    csv_rows(output/'damping_joint_radius_time.csv',joint)
    write_json(output/'run_summary.json',dict(evaluated=evaluate.evaluated,reused=evaluate.reused,
        statuses={model:dict(Counter(r['status'] for r in population if r['model_id']==model)) for model in (REPO_MODEL,XUE_MODEL)},
        disk_audit=disk_record))
    from damping_comparison_figures import make_figures, make_html_report
    make_figures(output,population,flux,fixed,tracks)
    make_report(output,metadata,population,flux,fixed,scan,sensitivities,boundaries,audit,disk_record)
    make_html_report(output)
    print(f'Wrote {output}; evaluated {evaluate.evaluated}, reused {evaluate.reused}',flush=True)


def make_report(output,metadata,population,flux,fixed,scan,sensitivities,boundaries,audit,disk_record):
    lines=['# Damping model comparison', '',
        'This is a first-stage damping experiment. It does not calibrate IMRI or LISA event counts.', '',
        '## Equations and physical interpretation', '',
        'Xue et al. (2026), Sec II.3 Eq (2)–(3), adopt the combined hydrodynamic fit from Whitehead et al. (2025), Eq (42):', '',
        '```text',
        'r_H = r (m / (3 M))^(1/3)',
        'm_tilde = 2 pi r_H^2 Sigma / m',
        'i_tilde = sin(i) / (H/r)',
        'f = |Delta i|/i = 10^1.80 m_tilde^0.67 max(i_tilde, 4.6)^(-2.64)',
        'di/dt = - i f / (P/2),  P = 2 pi sqrt(r^3/(GM))',
        '```', '',
        'Logarithms in the source fit are base 10. The physical derivative is negative; the fitted decrement is a positive magnitude. Two crossings occur per orbit. Xue Eq (8), p_disk=(2/pi) asin[(H/r)/sin(i)], is a geometric residence fraction. It is exposed by effective_crossing_probability, saturated to 1 inside the disk, and is not multiplied into the fitted damping rate a second time.', '',
        'The repository uses drag impulses at the two true orbital nodes r±=a(1-e²)/(1±e cos omega). Each active node samples the nearest solved disk cell; v_z=sqrt(GMa(1-e²)) sin(i)/r_node, t_cross=2H/v_z, r_drag=2Gm/v_rel² for BHs, t_drag=m/(pi r_drag² rho v_rel). The orbit-averaged osculating da/de/di equations are implemented in physical_process/damp.py and independently tested against Cartesian drag impulses in tests/test_damping_equations.py. They have no sound-speed regularization, Hill cap, or explicit stellar-encounter term.', '',
        'The native adaptive explicit Euler integrator controls fractional changes in a and i at 1%; this is not a local truncation-error estimator. The comparison adapter supplies stop_time_s at AGN shutdown and linearly interpolates the last accepted step to the first inclination threshold. Production solver code and defaults are unchanged. Low inclination alone does not qualify an eccentric orbit: e<=0.01 and radial containment are required. Aligned eccentric sources are retained as an unresolved handoff state.', '',
        '## Calibration and completeness', '',
        'The evaluated Xue subset uses the rounded Rowan et al. (2025), Sec 2.6 reference environment: M=10^7 Msun, m=25 Msun, r=0.01 pc, H/r=0.0057, Sigma0=1.6e4 g/cm², circular prograde sources up to 15°, with Sigma/Sigma0 in [0.25,2]. Its combined low-i branch comes from Whitehead et al.; transfer to this reference host is explicitly flagged. These are restricted applications of the combined published fit. Interpolation between the discrete density suites and transfer of its low-i branch are assumptions, not direct simulations of every joint point. Rowan evolves inviscid gas; alpha is not a calibrated validity condition. A rejected state means outside the verified operational subset, not necessarily outside every published simulation. Whitehead Fig 5 gives fitted low-i Hill mass ratios 0.005, 0.012, 0.023, 0.031, 0.058, 0.061, 0.169 and 0.199; the stalled 0.115 case is excluded. Full embedding predictions are restricted to 0.005<=m_tilde<=0.199 so the low-i branch is not extrapolated in Hill mass. In particular the Rowan 2Sigma0 state has m_tilde~0.341 and is unavailable for full embedding. The local fixture does not reproduce hydrodynamic vertical structure or evolving disk disturbances.', '',
        'Whitehead simulations started circular, prograde, at [1,2,5]H/R, M=4e6 Msun, radii [5e3,1e4,2e4] r_g and Eddington ratios [0.05,0.16,0.5]. The cold stalled environment was excluded from fitting. The combined coefficients have published uncertainties (0.67±0.02, -2.64±0.06, 1.80±0.09); log10(knee)=0.66±0.02 in the calibration paper, while Xue adopts 4.6. Combined scatter is about 0.2 dex. Bootstrap intervals here describe finite population sampling only; they do not include that physical calibration scatter or coefficient covariance.', '',
        'Xue supplies an inclination decrement, not an independently calibrated a/e/E/|L| evolution. The Xue adapter therefore holds reference radius and circular eccentricity fixed, flags that closure, and leaves total energy and angular-momentum changes blank. Its embedding radii are assumed reference radii, not predictions of radial transport. The repository evolves a/e/i and reports total Newtonian orbital E=-GMm/(2a), |L|=m sqrt(GMa(1-e²)); changes are final minus initial. Negative delta_E means the orbit becomes more bound; negative delta_|L| means magnitude loss. Gas can transfer energy at individual eccentric nodes; universal per-node energy loss is not assumed.', '',
        'direct_disk_damping is evaluated. direct_plus_effective_scattering is unavailable: no independent additive encounter calibration was found in the cited papers. Its inclination, energy and angular-momentum contributions remain missing. The repository relaxation timescale is diagnostic only. This benchmark includes no first-principles two-body relaxation.', '',
        '## Matched population and weighted first-embedding flux', '',
        f"Population: {metadata['source_population']}. Both models receive bit-identical serialized inputs, IDs, weights and seeds, checked by input SHA-256. No IMF/remnant mapping or population normalization is changed between models.", '',
        'For generated fixtures, each realization samples inclination uniformly over [2°,15°], log-uniform Sigma/Sigma0 over [0.25,1], and positive uniform raw weights over [0.5,1.5], normalized to the declared source count. Masses, radius and aspect ratio are fixed to the reference. This controlled distribution is not a production NSC population or a physical event-rate normalization. The density range was selected from the literature overlap before the full experiment; both models always receive the same cohort.', '',
        'For each realization and time bin, F_hat=sum(weights of first events)/Delta t, in weighted objects/Myr. The CSV reports equal-realization means and 95% percentile intervals from paired resampling of whole realizations. Cumulative fractions divide by the entire realization input weight. All censored, terminated, failed and unsupported sources are retained in damping_source_outcomes.csv. Unavailable data never become evaluated zero; incomplete full-population estimates and ratios are blank, with measured lower bounds separately named.', '',
        'A repo/Xue bin flux ratio requires complete coverage, positive fluxes and Kish effective event counts >=5 in both models. Zero, low_count and incomplete ratios have explicit statuses. Bins are left-inclusive/right-exclusive except the final right endpoint. Initial embedded sources count at time zero. No late-time extrapolation or survival extrapolation is performed.', '',
        '| Model | Embedded | Censored | Other | Final weighted embedded fraction (95% interval) |',
        '|---|---:|---:|---:|---|']
    for model,prefix in ((REPO_MODEL,'repo'),(XUE_MODEL,'xue')):
        subset=[r for r in population if r['model_id']==model]
        counts=Counter(r['status'] for r in subset)
        fraction=flux[-1][prefix+'_cumulative_fraction'];ci=flux[-1][prefix+'_cumulative_fraction_ci95']
        lines.append(f"| {model} | {counts[OutcomeStatus.EMBEDDED]} | {counts[OutcomeStatus.CENSORED]} | {len(subset)-counts[OutcomeStatus.EMBEDDED]-counts[OutcomeStatus.CENSORED]} | {fraction} {ci} |")
    lines += ['', '| Time bin (Myr) | Repo mean flux | Xue mean flux | Ratio | Ratio status |',
              '|---|---:|---:|---:|---|']
    for row in flux:
        lines.append(f"| {row['time_left_myr']:.4g}–{row['time_right_myr']:.4g} | {row['repo_flux']} | {row['xue_flux']} | {row['ratio']} | {row['ratio_status']} |")
    lines += ['', '## Fixed sources, radial histories and orbital losses', '',
              '| Initial i (deg) | Repo t_embed (Myr) | Xue t_embed (Myr) | Repo/Xue time | Repo r_final/r_initial | Repo delta E (erg) | Repo delta L_magnitude (g cm²/s) |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(0,len(fixed),2):
        a,b=fixed[j:j+2]
        ratio=a['t_embed_myr']/b['t_embed_myr'] if a['t_embed_myr'] and b['t_embed_myr'] else None
        lines.append(f"| {np.rad2deg(a['inclination_rad']):g} | {a['t_embed_myr']} | {b['t_embed_myr']} | {ratio} | {a['final_radius_cm']/a['radius_cm']:.8g} | {a['delta_energy_erg']} | {a['delta_angular_momentum_g_cm2_s']} |")
    lines += ['', 'damping_trajectories.csv contains time, inclination, eccentricity and radius along every fixed-source history and the pAGN audit. damping_joint_radius_time.csv contains weighted first-event time/radius histograms per realization. The source outcomes preserve unbinned first-passage distributions and every non-embedding source. For Xue, radial constancy is an assumption and E/|L| columns are missing.', '',
              '## Sensitivities and scan coverage', '',
              'damping_parameter_scan.csv varies inclination, semimajor axis, eccentricity, object mass, host mass, surface density, aspect ratio, viscosity and lifetime. Positive scale variables use logarithmic grids where appropriate; eccentricity includes zero and high-e regimes. Every requested point has both model rows, even when one is out of range.', '',
              'Local first-order logarithmic sensitivities use symmetric log steps ±0.03 about the reference. A derivative is withheld if either endpoint is unsupported or does not embed. log sensitivity at e=0 is undefined. The viscosity scan holds the local gas state fixed; alpha has no explicit term in either direct drag law, so zero sensitivity here is a structural property of the adapters, not a calibrated prediction about viscosity or self-consistent disk response to alpha. Rowan simulations are inviscid.', '',
              '| Parameter | Model | d ln t_embed / d ln parameter | Status |',
              '|---|---|---:|---|']
    for row in sensitivities:
        lines.append(f"| {row['parameter']} | {row['model_id']} | {row.get('dln_tembed_dln_parameter')} | {row['status']} |")
    lines += ['', '## Lifetime boundaries and existing-disk audit', '',
        'damping_lifetime_boundaries.csv tests shutdown at 0.001, 0.1, 0.999, 1, 1.001, 10 and 100 times each model’s reference first-embedding time. Both models use the identical deadline for each pair; events at the endpoint count as embedded. Numerical uncertainty near the repo Euler threshold is exposed by the 0.999/1.001 cases. The automated adapter test also tightens the native fractional-change cap by a factor of four and checks embedding-time/radius convergence.', '',
        f"Existing pAGN audit status: `{canonical(disk_record)}`.", '',
        'The existing Sirko profile, alpha=0.1, Eddington ratio=0.1 and Q=1 outer truncation, is saved with solved midplane density and sound speed. The audit places identical circular/eccentric compact objects at fractions of its physical outer radius. Xue states outside the verified subset remain unavailable. This domain audit prevents the circular local-fixture result from being presented as production-population agreement. No new disk family or migration prescription is implemented.', '',
        '## Findings and limits', '']
    paired=[(fixed[j],fixed[j+1]) for j in range(0,len(fixed),2) if fixed[j]['t_embed_myr'] and fixed[j+1]['t_embed_myr']]
    if paired:
        ratios=[a['t_embed_myr']/b['t_embed_myr'] for a,b in paired]
        lines.append(f"In the evaluated local fixtures the repo/Xue first-embedding time ratio ranges from {min(ratios):.4g} to {max(ratios):.4g}. These calculations do not support equality of the two damping prescriptions in this controlled subset. Their different inclination dependence and drag normalization produce different first-event timing. The cumulative fraction at a finite AGN lifetime can mask a strong timing discrepancy when both populations eventually embed.")
    lines += ['', 'The fit predicts |di/dt| proportional to Sigma^0.67 at fixed geometry; the repo BH drag is linear in solved density. At high inclination the fitted fractional decrement scales as sin(i)^(-2.64), then saturates below the knee. The repo node velocity, gravitational focusing and crossing duration produce a different dependence. Full population equivalence remains unresolved for eccentric, retrograde, other-mass and other-disk states, and radial-loss agreement cannot be established from an inclination-only fit. Repeated-realization intervals must not be interpreted as uncertainty on missing physics.', '',
        'The local fixture holds Sigma and H/r constant and sets rho=Sigma/(2hr) at every radial grid point. Its density therefore changes with radius; it is not a solved global accretion disk. This one-zone fixture and repository vertical averaging differ from stratified, disturbed hydrodynamic disks. BH drag near embedding is not regularized by sound speed in the existing code. These are plausible drivers to investigate, not fitted explanations. No parameters were tuned to force agreement, and no downstream event counts were evaluated.', '',
        '## Reproduction and provenance', '', '```bash',
        f"PYTHONPATH=src {metadata['executable']} {' '.join(sys.argv)}", '```', '',
        'Repeat with --resume to reuse atomic source checkpoints; complete input, model, experiment and code digests must match. A changed code/config/cohort requires a separate output directory. matched_inputs.json can be supplied to --input-json to replay the exact cohort in a new output directory. manifest.json records resolved settings, package versions, seed schedule and code/input hashes. Checkpoint files contain their full identities; outputs are written atomically.', '',
        'Validation command:', '', '```bash',
        f"PYTHONPATH=src {metadata['executable']} -m pytest tests/test_damping_models.py tests/test_damping_statistics.py tests/test_damping_equations.py tests/test_damping_adaptive.py", '```', '',
        '## Figures', '',
        '![Weighted flux, cumulative embedding, histories and radius](damping_comparison.png)', '',
        'The confidence bands are paired realization bootstrap intervals. The first flux bin is displayed at its positive arithmetic midpoint; all subsequent bins use geometric midpoints. Exact t=0 events remain in source tables; logarithmic trajectory plots cannot display the time-zero coordinate. Static figure: damping_comparison.png and damping_comparison.pdf.', '',
        '## Sources', '',
        '- Xue et al. (2026), Sec II.3 Eq (2)–(4), Eq (8): https://arxiv.org/abs/2605.29305',
        '- Whitehead, Rowan & Kocsis (2025), combined fit Eq (42), calibration Table 1/2: https://arxiv.org/abs/2505.23899',
        '- Rowan et al. (2025), reference environment Sec 2.6, inclination and density suites: https://arxiv.org/abs/2505.23739', '']
    (output/'damping_model_comparison_report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
