"""Tests that need neither XSPEC nor pocoMC."""

import math

import numpy as np
import pytest

from xspecsampler import Parameter, resolve_priors, resolve_profile, profile_hash, weighted_quantiles
from xspecsampler.diagnostics import compare_seeds, parameter_summary, summed_evidence
from xspecsampler.priors import log_prior_volume, to_coordinates, to_physical


def param(component, name, group=1, lo=0.0, hi=100.0):
    return Parameter(label=f"DG{group}:{component}.{name}", data_group=group, component=component, name=name,
                     xspec_index=1, local_index=1, value=1.0, delta=0.01, hard_min=lo, soft_min=lo,
                     soft_max=hi, hard_max=hi)


MANIFEST = {"profiles": {
    "standard": {"rules": [
        {"component": "powerlaw", "name": "PhoIndex", "distribution": "uniform", "min": 1.0, "max": 3.0},
        {"component": "powerlaw", "name": "norm", "distribution": "loguniform", "min": 1e-3, "max": 10.0},
        {"component": "constant", "name": "factor", "data_group": 2, "distribution": "uniform", "min": 0.9, "max": 1.1},
    ]},
    "narrow": {"inherits": "standard",
               "replace": [{"component": "powerlaw", "name": "PhoIndex", "distribution": "uniform", "min": 1.5, "max": 2.5}],
               "add": [{"component": "TBabs", "name": "nH", "distribution": "uniform", "min": 0.1, "max": 2.0}]},
}}


def test_profiles_inherit_replace_add():
    rules = resolve_profile(MANIFEST, "narrow")
    assert [r["min"] for r in rules if r["name"] == "PhoIndex"] == [1.5]
    assert any(r["component"] == "TBabs" for r in rules)
    assert profile_hash(MANIFEST, "narrow") != profile_hash(MANIFEST, "standard")


def test_profile_errors():
    bad = {"profiles": {"a": {"rules": []}, "b": {"inherits": "a", "replace": [{"component": "x", "name": "y"}]}}}
    with pytest.raises(ValueError):
        resolve_profile(bad, "b")
    dup = {"profiles": {"a": {"rules": [{"component": "x", "name": "y"}]}, "b": {"inherits": "a", "add": [{"component": "x", "name": "y"}]}}}
    with pytest.raises(ValueError):
        resolve_profile(dup, "b")


def test_resolve_priors_matching_and_limits():
    params = [param("powerlaw", "PhoIndex", lo=-3, hi=10), param("powerlaw", "norm", lo=0, hi=1e24),
              param("constant", "factor", group=2, lo=0, hi=1e10)]
    priors = resolve_priors(resolve_profile(MANIFEST, "standard"), params)
    assert [p.distribution for p in priors] == ["uniform", "loguniform", "uniform"]
    with pytest.raises(ValueError, match="matched 0"):
        resolve_priors(resolve_profile(MANIFEST, "standard"), [param("bbody", "kT")])
    with pytest.raises(ValueError, match="hard limits"):
        resolve_priors(resolve_profile(MANIFEST, "standard"), [param("powerlaw", "PhoIndex", lo=1.2, hi=10)])


def test_coordinate_transforms_round_trip():
    params = [param("powerlaw", "PhoIndex", lo=-3, hi=10), param("powerlaw", "norm", lo=0, hi=1e24)]
    priors = resolve_priors(resolve_profile(MANIFEST, "standard"), params)
    x = np.array([[2.0, 0.5], [1.5, 3.0]])
    assert np.allclose(to_physical(priors, to_coordinates(priors, x)), x)
    assert math.isclose(log_prior_volume(priors), math.log(2.0) + math.log(4.0))


def test_weighted_quantiles_and_railing():
    rng = np.random.default_rng(1)
    x = rng.uniform(1.0, 3.0, 20000)
    w = np.ones_like(x)
    assert abs(weighted_quantiles(x, w, [0.5])[0] - 2.0) < 0.02
    params = [param("powerlaw", "PhoIndex", lo=-3, hi=10)]
    priors = resolve_priors(resolve_profile(MANIFEST, "standard"), params)
    piled = np.concatenate([np.full(9000, 1.001), rng.uniform(1, 3, 1000)])[:, None]
    row = parameter_summary(piled, np.ones(len(piled)), priors)[0]
    assert row["railed_lower"] == 1 and row["railed_upper"] == 0


class _R:
    def __init__(self, z, ez, rows):
        self.log_evidence, self.log_evidence_error, self.summary = z, ez, rows


def test_compare_seeds_uses_quadrature_half_widths():
    a = _R(-100.0, 0.3, [{"label": "p", "p16": 0.0, "p50": 1.0, "p84": 2.0}])
    b = _R(-103.0, 0.4, [{"label": "p", "p16": 2.0, "p50": 3.0, "p84": 4.0}])
    out = compare_seeds(a, b)
    assert math.isclose(out["delta_log_evidence"], -3.0)
    assert math.isclose(out["delta_over_sigma"], 3.0 / 0.5)
    assert math.isclose(out["parameters"][0]["shift_over_width"], 2.0 / math.sqrt(2.0))
    total, err = summed_evidence([a, b])
    assert math.isclose(total, -203.0) and math.isclose(err, 0.5)


def test_checkpoint_order_and_gates():
    pytest.importorskip("scipy")
    from xspecsampler.sampler import checkpoint_key
    from xspecsampler.gates import QualityGates, failures

    names = ["poco_x_10.state", "poco_x_2.state", "poco_x_final.state", "junk.state"]
    assert sorted(names, key=checkpoint_key) == ["junk.state", "poco_x_2.state", "poco_x_10.state", "poco_x_final.state"]

    class Res:
        ess, log_evidence_error, best_statistic = 600.0, 0.5, 1010.0
    assert failures(Res, QualityGates(), reference_statistic=1000.0) == []
    assert len(failures(Res, QualityGates(), reference_statistic=980.0)) == 1
    Res.ess = 100.0
    assert any("ESS" in f for f in failures(Res, QualityGates()))
