"""End-to-end run on a small model: LM multistart, gated pocoMC posterior, polish.

Uses one NuSTAR FPMA spectrum from the reference directory
(XSPECSAMPLER_REFERENCE_ROOT, see test_reference_results.py) with an absorbed
power law over 20-60 keV, so it runs in a few minutes on one core.  Needs
PyXspec and pocoMC.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xspec")
pytest.importorskip("pocomc")
pytestmark = pytest.mark.xspec

import xspecsampler as xs  # noqa: E402

REF = Path(os.environ.get("XSPECSAMPLER_REFERENCE_ROOT", "/nonexistent"))
SRC = REF / "results" / "polish" / "fit_30001011002_m2_noline_constant_afefree_i27p5_a0p95_qfree_s0p01_aa779e835d"
if not SRC.is_dir():
    pytest.skip("reference results not available", allow_module_level=True)

XCM_TEXT = """statistic chi
data 1:1 outA.pha
backgrnd 1 nu30001011002A01_bk.pha
ignore 1:**-20.0 60.0-**
method leven 1000 0.01
abund wilm
xsect vern
model  TBabs*powerlaw
            0.6         -1
            2.2       0.01         -3         -2          9         10
              1       0.01          0          0      1e+20      1e+24
"""

PRIORS = {"profiles": {"pl": {"rules": [
    {"component": "powerlaw", "name": "PhoIndex", "distribution": "uniform", "min": 1.5, "max": 3.5},
    {"component": "powerlaw", "name": "norm", "distribution": "loguniform", "min": 1e-3, "max": 1e3},
]}}}

TINY = [("tiny", {"n_effective": 128, "n_active": 64, "n_total": 512, "n_evidence": 512,
                  "train_frequency": 1, "flow_epochs": 100, "flow_patience": 10, "save_every": 2})]


@pytest.fixture(scope="module")
def xcm(tmp_path_factory):
    work = tmp_path_factory.mktemp("pl")
    for name in ("outA.pha", "nu30001011002A01_bk.pha", "nu30001011002A01_sr.rmf", "nu30001011002A01_sr.arf"):
        shutil.copy(SRC / name, work / name)
    path = work / "pl.xcm"
    path.write_text(XCM_TEXT)
    return path


def test_end_to_end(xcm, tmp_path):
    info = xs.read_xcm(xcm)
    assert info.labels == ["DG1:powerlaw.PhoIndex", "DG1:powerlaw.norm"]

    starts = [{"source": "xcm"}, {"perturb_seed": 1, "source": "jitter"},
              {"start": {"DG1:powerlaw.PhoIndex": 3.0, "DG1:powerlaw.norm": 0.1}, "source": "far"}]
    results, failed = xs.multistart(xcm, starts, n_procs=2, save_best_xcm=tmp_path / "best.xcm")
    assert not failed and len(results) == 3
    best = results[0]
    assert best.statistic <= min(r.start_statistic for r in results)
    assert abs(xs.read_xcm(tmp_path / "best.xcm").statistic - best.statistic) < 1e-3

    priors = xs.resolve_priors(xs.resolve_profile(PRIORS, "pl"), info.parameters)
    result, history = xs.sample_with_gates(
        xcm, priors, tmp_path / "post", seed=7, tiers=TINY, n_procs=1,
        gates=xs.QualityGates(min_ess=100), reference_statistic=best.statistic,
    )
    assert history[0]["failures"] == []
    assert np.isfinite(result.log_evidence)
    assert result.best_statistic - best.statistic < 25
    loaded = xs.SampleResult.load(tmp_path / "post" / "tiny")
    assert loaded.labels == result.labels and np.allclose(loaded.weights, result.weights)

    # A second call reuses the finished tier instead of sampling again.
    again, _ = xs.sample_with_gates(xcm, priors, tmp_path / "post", seed=7, tiers=TINY, n_procs=1,
                                    gates=xs.QualityGates(min_ess=100), reference_statistic=best.statistic)
    assert again.log_evidence == result.log_evidence

    polished, candidates = xs.polish(xcm, [result], lm_result=best)
    assert polished.statistic <= best.statistic + 1e-6
    assert len(candidates) == 2
