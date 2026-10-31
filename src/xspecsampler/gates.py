"""Quality gates for posterior samples.

A posterior is accepted if its effective sample size is large enough, if the
uncertainty of its evidence is small enough, and if its best sample is not
much worse than a reference fit.  The reference is normally the best fit
from the minimizer.  A best sample that is much worse than the reference
indicates that the sampler has not found the main mode of the posterior.

A run that fails the gates is repeated with larger particle numbers, tier by
tier, until one passes (:func:`sample_with_gates`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._io import config_hash
from .sampler import DEFAULT_SETTINGS, SampleResult, sample

_TRAIN = {"train_frequency": 1, "flow_epochs": 1000, "flow_patience": 10, "save_every": 5}

# Default escalation: each tier doubles the particle numbers of the previous one.
DEFAULT_TIERS = [
    ("base", dict(DEFAULT_SETTINGS)),
    ("escalated", {"n_effective": 1024, "n_active": 512, "n_total": 4096, "n_evidence": 4096, **_TRAIN}),
    ("final", {"n_effective": 2048, "n_active": 1024, "n_total": 8192, "n_evidence": 8192, **_TRAIN}),
    ("extended", {"n_effective": 4096, "n_active": 2048, "n_total": 16384, "n_evidence": 16384, **_TRAIN}),
]


@dataclass(frozen=True)
class QualityGates:
    """Acceptance thresholds for a posterior sample.

    Parameters
    ----------
    min_ess : float, optional
        Minimum effective sample size.  Default is 512.
    max_log_evidence_error : float, optional
        Maximum uncertainty of the natural-log evidence.  Default is 1.
    max_worse_than_reference : float, optional
        Maximum amount by which the lowest statistic among the samples may
        exceed the reference statistic (e.g. in chi-squared).  Default is 25.
    """
    min_ess: float = 512
    max_log_evidence_error: float = 1.0
    max_worse_than_reference: float = 25.0  # in the fit statistic, e.g. delta chi^2


class QualityGateError(RuntimeError):
    """Raised when no tier passes the quality gates.

    Attributes
    ----------
    history : list of dict
        Diagnostics of every tier that was run.
    """

    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


def failures(result, gates, reference_statistic=None):
    """Check a posterior against the quality gates.

    Parameters
    ----------
    result : SampleResult
        Posterior to check.
    gates : QualityGates
        Thresholds.
    reference_statistic : float, optional
        Statistic of the reference fit.  If not given, the comparison with
        the reference is skipped.

    Returns
    -------
    list of str
        Description of each failed gate.  The list is empty if the posterior
        passes.
    """
    out = []
    if result.ess < gates.min_ess:
        out.append(f"ESS {result.ess:.0f} < {gates.min_ess}")
    if result.log_evidence_error > gates.max_log_evidence_error:
        out.append(f"lnZ error {result.log_evidence_error:.3f} > {gates.max_log_evidence_error}")
    if reference_statistic is not None:
        delta = result.best_statistic - float(reference_statistic)
        if delta > gates.max_worse_than_reference:
            out.append(f"best sample is {delta:.2f} worse than the reference fit")
    return out


def sample_with_gates(model, priors, out_dir, seed=42, tiers=None, gates=QualityGates(),
                      reference_statistic=None, **sample_kwargs):
    """Sample a posterior with increasing particle numbers until it passes the gates.

    Parameters
    ----------
    model : str, path-like or XspecLikelihood
        XCM file, or an open likelihood.
    priors : list of Prior
        One prior per thawed parameter, in the order of the XCM file.
    out_dir : str or path-like
        Output directory.  Each tier writes to ``out_dir/<tier name>``.
    seed : int, optional
        Random seed of the sampler.  Default is 42.
    tiers : list of (str, dict), optional
        Names and sampler settings of the tiers, in the order in which they
        are tried.  Default is :data:`DEFAULT_TIERS`.
    gates : QualityGates, optional
        Thresholds.  Default is ``QualityGates()``.
    reference_statistic : float, optional
        Statistic of the reference fit.  We recommend always providing it,
        since the other gates cannot detect a sampler that has missed the
        main mode.
    **sample_kwargs
        Further keyword arguments passed to :func:`sample`.

    Returns
    -------
    result : SampleResult
        Posterior of the first tier that passes.
    history : list of dict
        Diagnostics of every tier that was run.

    Raises
    ------
    QualityGateError
        If no tier passes.

    Notes
    -----
    A tier that has already been completed with the same settings and seed is
    read from disk instead of being run again, so that an interrupted call
    can simply be repeated.
    """
    tiers = DEFAULT_TIERS if tiers is None else tiers
    out_dir = Path(out_dir)
    history = []
    for name, settings in tiers:
        tier_dir = out_dir / name
        result = None
        if (tier_dir / "status.json").is_file():
            done = SampleResult.load(tier_dir)
            if (done.status.get("seed") == int(seed)
                    and config_hash(done.status.get("settings")) == config_hash({**DEFAULT_SETTINGS, **settings})):
                result = done
        if result is None:
            result = sample(model, priors, tier_dir, seed=seed, settings=settings,
                            reference_statistic=reference_statistic, **sample_kwargs)
        problems = failures(result, gates, reference_statistic)
        history.append({"tier": name, "failures": problems, "ess": result.ess,
                        "log_evidence": result.log_evidence, "log_evidence_error": result.log_evidence_error,
                        "best_statistic": result.best_statistic})
        if not problems:
            return result, history
    raise QualityGateError(f"no tier passed the quality gates: {history[-1]['failures']}", history)
