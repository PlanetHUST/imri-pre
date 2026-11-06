"""Unit-explicit fixture construction for the damping validation harness.

These controlled compact-object cohorts are not a replacement NSC IMF or a
remnant prescription. Production populations may be supplied as DampingInput
records without resampling or changing their weights.
"""
from dataclasses import replace

import numpy as np

from . import constants as c
from .agn import AGNProperties
from .damp import DampingCalculator
from .damping_models import (DampingInput, REFERENCE_RADIUS_CM, REFERENCE_H,
                             REFERENCE_SIGMA, make_local_repo_calculator)


def reference_source(**changes) -> DampingInput:
    x = DampingInput(0., REFERENCE_RADIUS_CM, np.deg2rad(8.), 0.,
        25*c.msun, 1.e7*c.msun, REFERENCE_SIGMA, REFERENCE_H*REFERENCE_RADIUS_CM,
        1.19e6, 0., c.Myr)
    return resolve_fixture(replace(x, **changes))


def resolve_fixture(x: DampingInput) -> DampingInput:
    """Circular reference relative speed; eccentric fixtures are diagnostic only.

    For eccentric sources the actual two node velocities are calculated by the
    repo solver; this scalar is only an input summary, not used in its drag.
    """
    vk = np.sqrt(c.G*x.smbh_mass_g/x.radius_cm)
    return replace(x, relative_velocity_cm_s=float(2*vk*np.sin(x.inclination_rad/2)))


def adapter_for_profile(x: DampingInput, profile=None):
    if profile is None:
        return make_local_repo_calculator(x)
    return DampingCalculator(AGNProperties.from_mass(x.smbh_mass_g/c.msun,
        x.disk_lifetime_s/c.Myr), False, profile.radius_cm, profile.surface_density_g_cm2,
        profile.aspect_ratio, profile.inner_radius_cm, profile.outer_radius_cm,
        density_g_cm3=profile.density_g_cm3)


def source_on_profile(x: DampingInput, profile) -> DampingInput:
    """Use the same nearest-cell convention as the production damping solver."""
    k = int(np.argmin(abs(profile.radius_cm-x.radius_cm)))
    return resolve_fixture(replace(x, surface_density_g_cm2=float(profile.surface_density_g_cm2[k]),
        scale_height_cm=float(profile.aspect_ratio[k]*x.radius_cm),
        sound_speed_cm_s=float(profile.sound_speed_cm_s[k]), disk_model_id=profile.model_name))
