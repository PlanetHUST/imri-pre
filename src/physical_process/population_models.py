"""Literature-based initial populations, with explicit normalization and scope.

Pan's broken spatial density is realized as an isotropic Kepler-tracer DF,
not by equating a spatial-radius CDF to a semimajor-axis CDF. This does not
reproduce Pan's cluster potential or Fokker-Planck evolution.
"""
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import constants as const
from .sampling import sample_salpeter

LEGACY_POPIII = "legacy_popiii"
PAN_NSC = "pan2021_nsc"
PAN_PAPER = "pan2021_paper"
XUE_BH = "xue2026_bh"
POPULATION_MODELS = (LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH)
PAN_BH_NUMBER_RATIO = 1e-3
PAN_STAR_MASS_MSUN = 1.0
PAN_BH_MASS_MSUN = 10.0
XUE_BH_MIN_MSUN = 5.0
XUE_BH_MAX_MSUN = 15.0
XUE_BH_SLOPE = 2.35


def pan_counts(m_smbh_msun, m_nsc_msun, *, paper_normalization=False):
    """Integer populations; M_NSC is TOTAL component mass in conditional mode.

    Pan's published normalization instead sets stellar mass alone to 20 M_bh.
    Integer rounding is below one star mass in the conditional total budget.
    """
    if m_smbh_msun <= 0 or m_nsc_msun <= 0:
        raise ValueError("Population mass inputs must be positive")
    if paper_normalization:
        nstar = int(np.floor(20 * m_smbh_msun / PAN_STAR_MASS_MSUN))
        nbh = int(np.rint(PAN_BH_NUMBER_RATIO * nstar))
    else:
        expected_stars = m_nsc_msun / (PAN_STAR_MASS_MSUN + PAN_BH_NUMBER_RATIO * PAN_BH_MASS_MSUN)
        nbh = int(np.rint(PAN_BH_NUMBER_RATIO * expected_stars))
        nstar = int(np.floor((m_nsc_msun - nbh * PAN_BH_MASS_MSUN) / PAN_STAR_MASS_MSUN))
    if nstar < 0:
        raise ValueError("Mass budget is too small for integer component allocation")
    return nstar, nbh


def published_masses(model, m_smbh_msun, m_nsc_msun):
    """Return object masses and BH-formation clocks [yr], not ZAMS lifetimes.

    Pre-existing BHs have clock 0; representative long-lived Pan stars have
    clock infinity and never automatically turn into BHs in this model.
    """
    if model in (PAN_NSC, PAN_PAPER):
        nstar, nbh = pan_counts(m_smbh_msun, m_nsc_msun, paper_normalization=model == PAN_PAPER)
        masses = np.concatenate((np.full(nstar, PAN_STAR_MASS_MSUN), np.full(nbh, PAN_BH_MASS_MSUN)))
        clocks = np.concatenate((np.full(nstar, np.inf), np.zeros(nbh)))
        order = np.random.permutation(len(masses))
        return masses[order], clocks[order]
    if model == XUE_BH:
        nbh = int(np.floor(m_smbh_msun / 20))
        return sample_salpeter(nbh, XUE_BH_MIN_MSUN, XUE_BH_MAX_MSUN, XUE_BH_SLOPE), np.zeros(nbh)
    raise ValueError("published_masses requires a published population preset")


def pan_scale_radius_cm(m_smbh_msun):
    """r_a=4 r_h, with Pan Eq. (19)'s M-sigma relation."""
    sigma_cm_s = 70e5 * (m_smbh_msun / 1.53e6)**(1 / 4.24)
    return 4 * const.G * m_smbh_msun * const.msun / sigma_cm_s**2


def tremaine_spatial_cdf(radius_cm, scale_radius_cm, gamma=1.5):
    radius = np.asarray(radius_cm, dtype=float)
    return (radius / (radius + scale_radius_cm))**(3 - gamma)


@lru_cache(maxsize=32)
def _kepler_tremaine_energy_cdf(gamma, maximum_log_energy):
    """Eddington inversion in dimensionless binding energy E/(GM/r_a).

    n(psi) is proportional to psi^4 (1+psi)^(gamma-4). In a Kepler
    potential, p(log E) is proportional to f(E) E^(-3/2). The positive
    second derivative below gives f(E) through Eddington's Abel integral.
    """
    if not 1 <= gamma < 3:
        raise ValueError("Tremaine Kepler tracer currently supports 1 <= gamma < 3")
    log_energy = np.linspace(-27.6310211159, maximum_log_energy, 4097)
    energy = np.exp(log_energy)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    theta = (nodes + 1) * np.pi / 4
    weights = weights * np.pi / 4
    psi = energy[:, None] * np.sin(theta)[None, :]**2
    second = psi**2 * (1 + psi)**(gamma - 6) * (12 + 8*(gamma - 1)*psi + gamma*(gamma - 1)*psi**2)
    distribution = 2 * np.sqrt(energy) * np.sum(second * np.sin(theta)[None, :] * weights[None, :], axis=1)
    log_energy_pdf = distribution * energy**(-1.5)
    cdf = np.concatenate(([0.0], cumulative_trapezoid(log_energy_pdf, log_energy)))
    cdf /= cdf[-1]
    return log_energy, cdf


def pan_orbits(count, m_smbh_msun, minimum_a_cm, gamma=1.5):
    """Full Tremaine spatial tracer profile, isotropic Kepler orbital sampling.

    a is not sampled from the spatial CDF. Thermal eccentricities and
    isotropic angles belong to this Kepler-tracer distribution. The stellar
    contribution to the gravitational potential is NOT included.
    """
    scale = pan_scale_radius_cm(m_smbh_msun)
    maximum_log_energy = float(np.log(scale / (2 * minimum_a_cm)))
    log_energy, cdf = _kepler_tremaine_energy_cdf(float(gamma), maximum_log_energy)
    binding = np.exp(np.interp(np.random.random(count), cdf, log_energy))
    a = scale / (2 * binding)
    eccentricity = np.sqrt(np.random.random(count))
    inclination = np.arccos(np.random.uniform(-1, 1, count))
    cos_omega = np.cos(np.random.uniform(0, 2*np.pi, count))
    return a, eccentricity, inclination, cos_omega, scale


def xue_radius_bounds_cm(m_smbh_msun):
    sigma_cm_s = 265e5 * (m_smbh_msun / 1e8)**.25
    gm = const.G * m_smbh_msun * const.msun
    minimum = (gm / (np.pi * 1e-4)**2)**(1/3)
    maximum = gm / sigma_cm_s**2
    return minimum, maximum


def xue_orbits(count, m_smbh_msun):
    """Mass/radial proxy only: uniform radius represented by circular orbits.

    Xue's detailed velocity prescription is not reproduced here. To avoid
    claiming r and a are interchangeable for eccentric orbits, this proxy
    fixes e=0, retains isotropic orientations, and says so in metadata.
    """
    minimum, maximum = xue_radius_bounds_cm(m_smbh_msun)
    a = np.random.uniform(minimum, maximum, count)
    eccentricity = np.zeros(count)
    inclination = np.arccos(np.random.uniform(-1, 1, count))
    cos_omega = np.cos(np.random.uniform(0, 2*np.pi, count))
    return a, eccentricity, inclination, cos_omega, maximum


def population_budget(model, m_smbh_msun, m_nsc_msun):
    """Deterministic component budgets / expectations for run provenance."""
    if model in (PAN_NSC, PAN_PAPER):
        nstar, nbh = pan_counts(m_smbh_msun, m_nsc_msun, paper_normalization=model == PAN_PAPER)
        total = nstar + nbh
        mass = nstar + 10 * nbh
        return dict(smbh_mass_msun=float(m_smbh_msun), nsc_mass_input_msun=float(m_nsc_msun),
                    initial_stars=nstar, initial_bhs=nbh, initial_objects=total,
                    actual_component_mass_msun=float(mass),
                    mean_object_mass_msun=mass / total if total else None,
                    pan_scale_radius_cm=pan_scale_radius_cm(m_smbh_msun))
    if model == XUE_BH:
        nbh = int(np.floor(m_smbh_msun / 20))
        alpha, low, high = XUE_BH_SLOPE, XUE_BH_MIN_MSUN, XUE_BH_MAX_MSUN
        mean = ((high**(2-alpha)-low**(2-alpha))/(2-alpha)) / ((high**(1-alpha)-low**(1-alpha))/(1-alpha))
        return dict(smbh_mass_msun=float(m_smbh_msun), nsc_mass_input_msun=float(m_nsc_msun),
                    initial_stars=0, initial_bhs=nbh, initial_objects=nbh,
                    expected_mean_object_mass_msun=mean, expected_component_mass_msun=nbh * mean,
                    actual_component_mass="random draw; see sampled population, not M_NSC")
    return dict(smbh_mass_msun=float(m_smbh_msun), nsc_mass_input_msun=float(m_nsc_msun),
                initial_preexisting_bhs=0, counts="sampled to total mass; see repeat results")


def population_description(model):
    if model in (PAN_NSC, PAN_PAPER):
        return dict(
            model=model, record_contract="preexisting_component_clocks_and_labels_v1",
            star_mass_msun=1.0, bh_mass_msun=10.0,
            bh_to_star_number_ratio=PAN_BH_NUMBER_RATIO,
            normalization=("M_NSC is total star+BH mass" if model == PAN_NSC else "M_star=20*M_bullet; M_NSC input unused"),
            nsc_mass_input_used=model == PAN_NSC,
            radial_spatial_density="n(r) proportional to r^(-gamma)*(r+r_a)^(gamma-4)",
            radial_scale="r_a=4*G*M_bullet/sigma^2; sigma=70 km/s*(M_bullet/1.53e6 Msun)^(1/4.24)",
            radial_sampling="isotropic Eddington-inverted Kepler tracer, not spatial CDF assigned to a",
            angular_sampling="isotropic cos(i), uniform omega, thermal eccentricity",
            scope="published component masses, abundance and full spatial tracer shape; Kepler-only potential, not Pan's cluster potential or Fokker-Planck evolution",
            empirical_nsc_radius_input_used=False,
        )
    if model == XUE_BH:
        return dict(model=model, record_contract="preexisting_component_clocks_and_labels_v1",
            bh_mass_min_msun=5.0, bh_mass_max_msun=15.0, bh_mass_slope=2.35,
            normalization="N_BH=floor(M_bullet/(20 Msun)); M_NSC input unused", nsc_mass_input_used=False,
            radial_number_density="dN/dr proportional to r^0; circular-radius proxy only",
            equivalent_spherical_density="n(r) proportional to r^-2 (NOT a uniform spatial density)",
            radial_bounds="source f_GW=1e-4 Hz to G*M_bullet/sigma^2, sigma=265 km/s*(M_bullet/1e8 Msun)^0.25",
            angular_sampling="circular and isotropic proxy; NOT Xue's beta_v=0.2 Gaussian velocity prescription",
            scope="mass, number normalization and radial-law control; no accretion, mergers, recoil or published kinematics",
            empirical_nsc_radius_input_used=False)
    return dict(model=LEGACY_POPIII, normalization="sample analytic IMF to M_NSC",
        nsc_mass_input_used=True, empirical_nsc_radius_input_used=True,
        radial_sampling="legacy p(a) proportional to a^(2-gamma), truncated at empirical NSC radius",
        angular_sampling="legacy thermal eccentricity, uniform i from 10 to 170 degrees, uniform cos(omega) from 0 to 1")
