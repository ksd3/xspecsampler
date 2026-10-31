"""Posterior summaries and seed-agreement diagnostics."""

from __future__ import annotations

import math

import numpy as np


def weighted_quantiles(values, weights, quantiles):
    """Compute quantiles of a weighted sample.

    Parameters
    ----------
    values : array_like
        Sample values.
    weights : array_like
        Non-negative weights, one per value.  They need not be normalized.
    quantiles : sequence of float
        Quantiles to compute, each in [0, 1].

    Returns
    -------
    list of float
        The requested quantiles, obtained by linear interpolation of the
        weighted empirical distribution function.
    """
    order = np.argsort(values)
    x = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(w)
    cumulative /= cumulative[-1]
    return [float(np.interp(q, cumulative, x)) for q in quantiles]


def effective_sample_size(weights):
    r"""Compute the effective sample size of a set of importance weights.

    Parameters
    ----------
    weights : array_like
        Non-negative weights.  They need not be normalized.

    Returns
    -------
    float
        Kish's effective sample size, :math:`1 / \sum_i w_i^2` for
        normalized weights :math:`w_i`.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def parameter_summary(samples, weights, priors):
    """Summarize a weighted posterior parameter by parameter.

    Parameters
    ----------
    samples : array_like, shape (n_samples, n_parameters)
        Posterior samples in physical units.
    weights : array_like, shape (n_samples,)
        Normalized posterior weights.
    priors : sequence of Prior
        Priors of the parameters, in the column order of `samples`.

    Returns
    -------
    list of dict
        One dictionary per parameter with the keys ``label``,
        ``prior_distribution``, ``prior_min``, ``prior_max``, the weighted
        percentiles ``p05``, ``p16``, ``p50``, ``p84`` and ``p95``,
        ``error_lower`` and ``error_upper`` (the 16th and 84th percentiles
        relative to the median), ``boundary_mass_lower_1pct`` and
        ``boundary_mass_upper_1pct``, and the flags ``railed_lower`` and
        ``railed_upper``.

    Notes
    -----
    A parameter is considered railed at a prior bound if its 5th (or 95th)
    weighted percentile lies within 2% of the prior width of that bound.  The
    boundary masses are the posterior weight within 1% of the prior width of
    each bound.  Both are measured in the sampling coordinate, i.e. in log10
    for log-uniform priors.
    """
    samples = np.asarray(samples, dtype=float)
    rows = []
    for j, prior in enumerate(priors):
        x = samples[:, j]
        p05, p16, p50, p84, p95 = weighted_quantiles(x, weights, (0.05, 0.16, 0.50, 0.84, 0.95))
        coordinate = np.log10(x) if prior.is_log else x
        lo, hi = prior.coordinate_bounds()
        width = hi - lo
        q05c, q95c = weighted_quantiles(coordinate, weights, (0.05, 0.95))
        rows.append({
            "label": prior.label,
            "prior_distribution": prior.distribution,
            "prior_min": prior.min,
            "prior_max": prior.max,
            "p05": p05, "p16": p16, "p50": p50, "p84": p84, "p95": p95,
            "error_lower": p50 - p16,
            "error_upper": p84 - p50,
            "boundary_mass_lower_1pct": float(np.sum(np.asarray(weights)[coordinate <= lo + 0.01 * width])),
            "boundary_mass_upper_1pct": float(np.sum(np.asarray(weights)[coordinate >= hi - 0.01 * width])),
            "railed_lower": int((q05c - lo) <= 0.02 * width),
            "railed_upper": int((hi - q95c) <= 0.02 * width),
        })
    return rows


def compare_seeds(result_a, result_b):
    """Compare two posteriors of the same model obtained with different seeds.

    Parameters
    ----------
    result_a, result_b : SampleResult
        Posteriors of the same model and priors.  Any object with the
        attributes ``log_evidence``, ``log_evidence_error`` and ``summary``
        is accepted.

    Returns
    -------
    dict
        ``delta_log_evidence`` (b minus a), ``combined_sigma`` (the two
        evidence uncertainties added in quadrature), ``delta_over_sigma``,
        and ``parameters``, a list with one entry per parameter giving the
        shift of the median (``delta_p50``), the combined 68% half-width
        (``combined_half_width``) and their ratio (``shift_over_width``).

    Notes
    -----
    The 68% half-width of each posterior is half the distance between its
    16th and 84th percentiles; the two half-widths are added in quadrature.
    """
    dz = result_b.log_evidence - result_a.log_evidence
    sigma = math.hypot(result_a.log_evidence_error, result_b.log_evidence_error)
    by_label = {row["label"]: row for row in result_b.summary}
    params = []
    for row in result_a.summary:
        other = by_label[row["label"]]
        # 68% half-widths of the two posteriors, added in quadrature
        half = math.hypot(0.5 * (row["p84"] - row["p16"]), 0.5 * (other["p84"] - other["p16"]))
        shift = other["p50"] - row["p50"]
        if half > 0:
            tension = abs(shift) / half
        else:
            tension = 0.0 if shift == 0 else math.inf
        params.append({
            "label": row["label"],
            "delta_p50": shift,
            "combined_half_width": half,
            "shift_over_width": tension,
        })
    return {
        "delta_log_evidence": dz,
        "combined_sigma": sigma,
        "delta_over_sigma": abs(dz) / sigma if sigma > 0 else math.inf,
        "parameters": params,
    }


def summed_evidence(results):
    """Sum the log-evidences of independent data sets.

    Parameters
    ----------
    results : iterable of SampleResult
        Posteriors of independent data sets, all computed with the same
        fixed value of a shared parameter.

    Returns
    -------
    total : float
        Sum of the natural-log evidences.
    error : float
        The reported evidence uncertainties added in quadrature.

    Notes
    -----
    For a parameter that is common to all data sets, such as an elemental
    abundance, the joint posterior on a grid of fixed values is proportional
    to the prior times the product of the individual evidences.  Comparing
    the summed log-evidences across the grid therefore gives the joint
    constraint on the shared parameter.  The scatter between runs with
    different seeds is often larger than the reported uncertainties, and we
    recommend estimating it separately.
    """
    total = sum(r.log_evidence for r in results)
    error = math.sqrt(sum(r.log_evidence_error ** 2 for r in results))
    return total, error
