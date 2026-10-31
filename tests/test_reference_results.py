"""Regression checks against a stored set of reference results.

The reference directory holds LM fits (results/polish/<fit>/), posterior
samples (results/samples/<sample>/) and a seed-agreement table
(validation/seed_parameter_diagnostics.csv) for a relxill model.  Set
XSPECSAMPLER_REFERENCE_ROOT to that directory and RELXILL_PATH to the relxill
build; the tests are skipped when either is missing.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

xspec = pytest.importorskip("xspec")
pytestmark = pytest.mark.xspec

import xspecsampler as xs  # noqa: E402

REF = Path(os.environ.get("XSPECSAMPLER_REFERENCE_ROOT", "/nonexistent"))
RELXILL = os.environ.get("RELXILL_PATH")
FIT = "fit_30001011002_m2_noline_constant_afefree_i27p5_a0p95_qfree_s0p01_aa779e835d"
SAMPLE = "sample_30001011002_m2_noline_constant_afefree_i27p5_a0p95_qfree_s0p01_aa779e835d_standard_seed42_*"

if not (REF / "results" / "polish" / FIT / "fit.xcm").is_file() or not RELXILL:
    pytest.skip("reference results or RELXILL_PATH not available", allow_module_level=True)

XCM = REF / "results" / "polish" / FIT / "fit.xcm"
META = json.loads((REF / "results" / "polish" / FIT / "params.json").read_text())
LMOD = {"relxill": RELXILL}


@pytest.fixture(scope="module")
def info():
    return xs.read_xcm(XCM, lmod=LMOD)


def test_thawed_parameters_match_reference(info):
    expected = [(f"DG{r['data_group']}:{r['component']}.{r['name']}", int(r["xspec_index"])) for r in META["thawed_params"]]
    assert [(p.label, p.xspec_index) for p in info.parameters] == expected
    assert abs(info.statistic - META["chi2"]) < 5e-5 * META["chi2"]


def test_likelihood_reproduces_stored_posterior_likelihoods(info):
    sample_dir = next((REF / "results" / "samples").glob(SAMPLE))
    z = np.load(sample_dir / "posterior_samples.npz")
    assert list(z["parameter_labels"].astype(str)) == info.labels
    rng = np.random.default_rng(0)
    pick = rng.choice(len(z["samples"]), size=6, replace=False)
    with xs.XspecLikelihood(XCM, n_procs=2, lmod=LMOD, info=info) as like:
        got = like.log_likelihood(z["samples"][pick])
    # Same %e parameter transport and 10-digit statistic as the backend that wrote the reference.
    assert np.allclose(got, z["log_likelihood"][pick], rtol=0, atol=1e-6)


def test_fit_replays_best_fit_and_does_not_worsen(info):
    best = {p.label: p.value for p in info.parameters}
    replay = xs.fit(XCM, start=best, lmod=LMOD, perform=False)
    assert abs(replay.statistic - META["chi2"]) < 5e-5 * META["chi2"]
    refit = xs.fit(XCM, start=best, lmod=LMOD)
    assert refit.statistic <= replay.statistic + 1e-6


def test_seed_agreement_matches_reference_table():
    rows = [r for r in csv.DictReader(open(REF / "validation" / "seed_parameter_diagnostics.csv")) if r["fit_id"] == FIT]
    dirs = {r.split("_seed")[1].split("_")[0]: r for r in os.listdir(REF / "results" / "samples")
            if r.startswith(f"sample_{FIT[4:]}_standard_seed")}
    a = xs.SampleResult.load_legacy(REF / "results" / "samples" / dirs["42"])
    b = xs.SampleResult.load_legacy(REF / "results" / "samples" / dirs["4242"])
    ours = {p["label"]: p for p in xs.compare_seeds(a, b)["parameters"]}
    for row in rows:
        label = f"DG{row['data_group']}:{row['component']}.{row['name']}"
        assert np.isclose(ours[label]["shift_over_width"], float(row["absolute_delta_over_combined_width"]), rtol=1e-9)
