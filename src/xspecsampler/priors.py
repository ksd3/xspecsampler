"""Declarative priors for XSPEC parameters.

A prior manifest is JSON of the form::

    {"profiles": {
        "standard": {"rules": [
            {"component": "zxipcf", "name": "Nh", "distribution": "loguniform", "min": 0.05, "max": 50},
            {"component": "constant", "name": "factor", "data_group": 2,
             "distribution": "uniform", "min": 0.9, "max": 1.1}]},
        "wider_norms": {"inherits": "standard",
                        "replace": [...], "add": [...]}}}

Each thawed parameter must match exactly one rule.  The optional
``data_group`` key restricts a rule to one data group.  A profile may inherit
the rules of another profile, replace some of them and add new ones.

Since Bayesian evidences depend on the prior volume, evidences are only
comparable between runs that use the same priors.  The priors should
therefore be fixed before fitting rather than derived from a fit result, and
we recommend storing the value of :func:`profile_hash` with each result.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from ._io import config_hash

DISTRIBUTIONS = ("uniform", "loguniform")


def _key(rule):
    return (rule["component"], rule["name"], rule.get("data_group"))


def resolve_profile(manifest, name):
    """Resolve a prior profile into a flat list of rules.

    Parameters
    ----------
    manifest : dict
        Prior manifest with a ``profiles`` mapping (see the module docstring).
    name : str
        Name of the profile.

    Returns
    -------
    list of dict
        Rules of the profile, with inheritance, replacements and additions
        applied.

    Raises
    ------
    KeyError
        If the profile, or a profile it inherits from, does not exist.
    ValueError
        If a replacement does not match a rule of the base profile, or an
        added rule duplicates an existing one.
    """
    profiles = manifest["profiles"]
    if name not in profiles:
        raise KeyError(f"unknown prior profile {name!r}")
    profile = copy.deepcopy(profiles[name])
    if "inherits" not in profile:
        return profile["rules"]
    base = resolve_profile(manifest, profile["inherits"])
    replacements = profile.get("replace", [])
    used, out = set(), []
    for rule in base:
        match = next((i for i, r in enumerate(replacements) if _key(r) == _key(rule)), None)
        if match is not None:
            used.add(match)
        out.append(copy.deepcopy(replacements[match] if match is not None else rule))
    if used != set(range(len(replacements))):
        raise ValueError(f"profile {name!r} replaces rules that its base does not have")
    keys = {_key(r) for r in out}
    for rule in profile.get("add", []):
        if _key(rule) in keys:
            raise ValueError(f"profile {name!r} adds a duplicate rule for {_key(rule)}")
        keys.add(_key(rule))
        out.append(copy.deepcopy(rule))
    return out


def profile_hash(manifest, name):
    """Compute a hash that identifies a resolved prior profile.

    Parameters
    ----------
    manifest : dict
        Prior manifest.
    name : str
        Name of the profile.

    Returns
    -------
    str
        SHA-256 hash of the canonical JSON form of the resolved rules.  Two
        profiles have the same hash if and only if they resolve to the same
        rules.
    """
    return config_hash(resolve_profile(manifest, name))


@dataclass(frozen=True)
class Prior:
    """Prior of a single parameter.

    Attributes
    ----------
    label : str
        Parameter label, ``DG<group>:<component>.<name>``.
    distribution : {'uniform', 'loguniform'}
        Prior distribution.
    min, max : float
        Bounds of the prior.
    """
    label: str
    distribution: str
    min: float
    max: float

    @property
    def is_log(self):
        """bool: Whether the prior is log-uniform."""
        return self.distribution == "loguniform"

    def coordinate_bounds(self):
        """Return the prior bounds in the sampling coordinate.

        Returns
        -------
        tuple of float
            ``(min, max)`` for uniform priors and ``(log10(min), log10(max))``
            for log-uniform priors.
        """
        return (math.log10(self.min), math.log10(self.max)) if self.is_log else (self.min, self.max)


def resolve_priors(rules, parameters):
    """Assign a prior to each thawed parameter.

    Parameters
    ----------
    rules : list of dict
        Prior rules, for example from :func:`resolve_profile`.
    parameters : sequence of Parameter
        Thawed parameters, as returned by :func:`xspecsampler.read_xcm`.

    Returns
    -------
    list of Prior
        One prior per parameter, in the order of `parameters`.

    Raises
    ------
    ValueError
        If a parameter matches no rule or more than one rule, if a prior
        interval is invalid, if a log-uniform prior has a non-positive lower
        bound, or if a prior extends beyond the XSPEC hard limits of its
        parameter.
    """
    out = []
    for p in parameters:
        matches = [
            r for r in rules
            if r["component"] == p.component and r["name"] == p.name
            and (r.get("data_group") is None or int(r["data_group"]) == int(p.data_group))
        ]
        if len(matches) != 1:
            raise ValueError(f"{p.label} matched {len(matches)} prior rules; expected exactly one")
        rule = matches[0]
        lo, hi, dist = float(rule["min"]), float(rule["max"]), rule["distribution"]
        if dist not in DISTRIBUTIONS:
            raise ValueError(f"unsupported prior distribution {dist!r} for {p.label}")
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"invalid prior interval [{lo}, {hi}] for {p.label}")
        if lo < p.hard_min - 1e-10 or hi > p.hard_max + 1e-10:
            raise ValueError(f"prior [{lo}, {hi}] for {p.label} exceeds the XSPEC hard limits [{p.hard_min}, {p.hard_max}]")
        if dist == "loguniform" and lo <= 0:
            raise ValueError(f"log-uniform prior for {p.label} needs a positive lower bound")
        out.append(Prior(p.label, dist, lo, hi))
    return out


def to_coordinates(priors, values):
    """Transform physical parameter values to sampling coordinates.

    Parameters
    ----------
    priors : sequence of Prior
        Priors of the parameters.
    values : array_like, shape (..., n_parameters)
        Parameter values in physical units.

    Returns
    -------
    numpy.ndarray
        Values in sampling coordinates (log10 for log-uniform priors).
    """
    values = np.array(values, dtype=float, copy=True)
    for j, prior in enumerate(priors):
        if prior.is_log:
            values[..., j] = np.log10(values[..., j])
    return values


def to_physical(priors, coordinates):
    """Transform sampling coordinates to physical parameter values.

    Parameters
    ----------
    priors : sequence of Prior
        Priors of the parameters.
    coordinates : array_like, shape (..., n_parameters)
        Values in sampling coordinates.

    Returns
    -------
    numpy.ndarray
        Parameter values in physical units.  This is the inverse of
        :func:`to_coordinates`.
    """
    values = np.array(coordinates, dtype=float, copy=True)
    for j, prior in enumerate(priors):
        if prior.is_log:
            values[..., j] = np.power(10.0, values[..., j])
    return values


def log_prior_volume(priors):
    """Compute the natural logarithm of the prior volume.

    Parameters
    ----------
    priors : sequence of Prior
        Priors of the parameters.

    Returns
    -------
    float
        Logarithm of the product of the prior widths, measured in sampling
        coordinates.  This quantity is useful when comparing evidences
        obtained with different priors.
    """
    return float(sum(math.log(b - a) for a, b in (p.coordinate_bounds() for p in priors)))
