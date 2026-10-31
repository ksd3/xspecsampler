r"""Orbital phases from a linear ephemeris.

The phase at time :math:`t` is :math:`\phi = ((t - T_0)/P) \bmod 1`, where
:math:`P` is the orbital period and :math:`T_0` the reference epoch.  Times
should be given in the same system as the ephemeris (e.g. HJD or BJD expressed
as MJD); for orbital periods of days the difference between these systems is
negligible compared with the precision of most ephemerides.
"""

from __future__ import annotations

import math

import numpy as np


def orbital_phase(mjd, period, t0):
    """Compute the orbital phase at one or more times.

    Parameters
    ----------
    mjd : float or array_like
        Time(s) in MJD.
    period : float
        Orbital period in days.
    t0 : float
        Reference epoch (phase zero) in MJD.

    Returns
    -------
    numpy.ndarray or float
        Orbital phase in the interval [0, 1).

    Examples
    --------
    >>> round(float(orbital_phase(41874.207 + 1.5 * 5.599829, 5.599829, 41874.207)), 6)
    0.5
    """
    return np.mod((np.asarray(mjd, dtype=float) - t0) / period, 1.0)


def phase_interval(start, end, period, t0):
    """Compute the orbital phase interval covered by an observation.

    Parameters
    ----------
    start, end : float
        Start and end of the observation in MJD.
    period : float
        Orbital period in days.
    t0 : float
        Reference epoch (phase zero) in MJD.

    Returns
    -------
    phase_start, phase_end, phase_mid : float
        Phases at the start, end and midpoint of the observation, in [0, 1).
    span : float
        Fraction of an orbit covered by the observation.

    Raises
    ------
    ValueError
        If `end` is earlier than `start`.

    Notes
    -----
    An interval that crosses phase zero has ``phase_end < phase_start``.
    """
    span = (end - start) / period
    if span < 0:
        raise ValueError("end is before start")
    p0 = float(orbital_phase(start, period, t0))
    return p0, (p0 + span) % 1.0, (p0 + span / 2) % 1.0, span


def phase_distance(phase, reference=0.0):
    """Compute the shortest distance between two orbital phases.

    Parameters
    ----------
    phase : float
        Orbital phase.
    reference : float, optional
        Reference phase.  Default is 0.

    Returns
    -------
    float
        Phase difference in the interval [0, 0.5].
    """
    d = abs((phase - reference) % 1.0)
    return min(d, 1.0 - d)


def propagated_phase_error(mjd, period, t0, sigma_period, sigma_t0):
    """Compute the phase uncertainty implied by the ephemeris uncertainties.

    Parameters
    ----------
    mjd : float
        Time in MJD.
    period : float
        Orbital period in days.
    t0 : float
        Reference epoch in MJD.
    sigma_period : float
        1-sigma uncertainty of the period in days.
    sigma_t0 : float
        1-sigma uncertainty of the reference epoch in days.

    Returns
    -------
    float
        1-sigma uncertainty of the orbital phase at `mjd`.

    Notes
    -----
    The uncertainty grows linearly with the number of cycles elapsed since
    `t0`, and the two contributions are added in quadrature.
    """
    n = (mjd - t0) / period
    return math.hypot(n * sigma_period, sigma_t0) / period
