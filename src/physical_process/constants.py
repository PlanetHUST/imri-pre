"""Physical constants and project-wide unit conversions.

The simulation uses cgs internally: grams, centimeters, and seconds. Names
ending in ``_MYR``, ``_CM``, ``_G``, or ``_S`` make those units explicit.
Lower-case aliases are retained for compatibility with the existing modules.
"""

HUBBLE_TIME_MYR = 13.6e3
PARSEC_CM = 3.08568e18
SOLAR_MASS_G = 1.98892e33
ASTRONOMICAL_UNIT_CM = 1.495978707e13
SOLAR_RADIUS_CM = 6.95700e10
GRAVITATIONAL_CONSTANT_CGS = 6.6743015e-8
SPEED_OF_LIGHT_CM_S = 2.99792458e10
YEAR_S = 3.1557600e7
MEGAYEAR_S = 1.0e6 * YEAR_S

SPEED_OF_LIGHT_FIFTH_CGS = SPEED_OF_LIGHT_CM_S**5
GRAVITATIONAL_CONSTANT_CUBED_CGS = GRAVITATIONAL_CONSTANT_CGS**3

# Star-cluster model parameter: central density / half-mass density.
CENTRAL_TO_HALF_MASS_DENSITY_RATIO = 20.0

# Backward-compatible aliases used by the current simulation.
tHubble = HUBBLE_TIME_MYR
parsec = PARSEC_CM
msun = SOLAR_MASS_G
AU = ASTRONOMICAL_UNIT_CM
Rsun = SOLAR_RADIUS_CM
G = GRAVITATIONAL_CONSTANT_CGS
c = SPEED_OF_LIGHT_CM_S
c5 = SPEED_OF_LIGHT_FIFTH_CGS
G3 = GRAVITATIONAL_CONSTANT_CUBED_CGS
yr = YEAR_S
Myr = MEGAYEAR_S
c2hm_dens = CENTRAL_TO_HALF_MASS_DENSITY_RATIO
