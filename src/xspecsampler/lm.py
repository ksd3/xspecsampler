"""Fits with the Levenberg-Marquardt minimizer of XSPEC.

The module provides single fits (:func:`fit`), parallel fits from several
starting points (:func:`multistart`) and fits started from posterior samples
(:func:`polish`).

Every fit runs in its own XSPEC subprocess, starting from the restored XCM
file.  A fit never returns a worse statistic than that of its starting point,
and the best-fit vector is evaluated again before it is returned.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .xcm import run_worker


@dataclass
class FitResult:
    """Result of a single fit.

    Attributes
    ----------
    statistic : float
        Best fit statistic, i.e. the smaller of `start_statistic` and
        `fit_statistic`.
    dof : int
        Degrees of freedom.
    values : dict
        Best-fit values of the thawed parameters, keyed by label.
    start_statistic : float
        Statistic at the starting point.
    fit_statistic : float
        Statistic returned by the minimizer.
    start : dict
        Starting values that were requested.
    perturb_seed : int or None
        Seed of the random perturbation of the start, if any.
    source : str
        Free-form description of the start.
    labels : list of str
        Labels of the thawed parameters, in XSPEC order.
    """
    statistic: float
    dof: int
    values: dict
    start_statistic: float
    fit_statistic: float
    start: dict = field(default_factory=dict)
    perturb_seed: int | None = None
    source: str = ""
    labels: list = field(default_factory=list)

    @property
    def reduced(self):
        """float: Reduced statistic, or NaN if there are no degrees of freedom."""
        return self.statistic / self.dof if self.dof > 0 else float("nan")

    def vector(self, labels=None):
        """Return the best-fit values as an array.

        Parameters
        ----------
        labels : list of str, optional
            Order of the parameters.  Default is :attr:`labels`.

        Returns
        -------
        numpy.ndarray
            Best-fit values.
        """
        return np.array([self.values[label] for label in (labels or self.labels)])


def fit(xcm, start=None, perturb_seed=None, perturb_scale=0.05, lmod=None, save_xcm=None,
        n_iterations=1000, strict_start=True, source="", timeout=None, perform=True):
    """Fit a model with the XSPEC minimizer.

    Parameters
    ----------
    xcm : str or path-like
        XCM file containing the data and the model.
    start : dict, optional
        Starting values, keyed by parameter label
        (``DG<group>:<component>.<name>``).  Values outside the hard limits
        are clipped to them.  Parameters that are not listed keep their
        values from the XCM file.
    perturb_seed : int, optional
        If given, the start is perturbed by Gaussian noise drawn with this
        seed.  See Notes.
    perturb_scale : float, optional
        Width of the perturbation as a fraction of the soft parameter range.
        Default is 0.05.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
    save_xcm : str or path-like, optional
        If given, the best fit is saved to this XCM file.
    n_iterations : int, optional
        Maximum number of minimizer iterations.  Default is 1000.
    strict_start : bool, optional
        If True (default), raise an error when `start` contains labels of
        parameters that are not thawed.  If False, such labels are ignored.
    source : str, optional
        Free-form description of the start, stored in the result.
    timeout : float, optional
        Maximum run time of the XSPEC process in seconds.
    perform : bool, optional
        If False, the starting point is set and evaluated without fitting.
        This is useful to evaluate or save a given parameter vector.
        Default is True.

    Returns
    -------
    FitResult
        Result of the fit.

    Raises
    ------
    RuntimeError
        If XSPEC fails, if `start` contains unknown labels and
        `strict_start` is True, or if the statistic of the best-fit vector
        cannot be reproduced.

    Notes
    -----
    The perturbation of each parameter has a standard deviation of
    ``max(5 * |delta|, perturb_scale * (soft_max - soft_min))``, where
    ``delta`` is the XSPEC fit step size.  The perturbed values are clipped
    to the soft limits.

    If the saved XCM file is not in the same directory as the data, its first
    line records the data directory, so that the file can be restored by this
    package from any location.
    """
    job = {
        "xcm": str(Path(xcm).resolve()),
        "lmod": dict(lmod or {}),
        "start": {k: float(v) for k, v in (start or {}).items()},
        "perturb_seed": perturb_seed,
        "perturb_scale": perturb_scale,
        "save_xcm": str(Path(save_xcm).resolve()) if save_xcm else None,
        "n_iterations": n_iterations,
        "strict_start": strict_start,
        "fit": bool(perform),
    }
    r = run_worker("fit", stdin=json.dumps(job), timeout=timeout)
    return FitResult(
        statistic=r["statistic"], dof=r["dof"], values=r["values"],
        start_statistic=r["start_statistic"], fit_statistic=r["fit_statistic"],
        start=job["start"], perturb_seed=perturb_seed, source=source, labels=r["labels"],
    )


def multistart(xcm, starts, n_procs=1, lmod=None, save_best_xcm=None, **fit_kwargs):
    """Fit a model from several starting points in parallel.

    Parameters
    ----------
    xcm : str or path-like
        XCM file containing the data and the model.
    starts : list of dict or None
        One entry per fit, with the optional keys ``start``, ``perturb_seed``
        and ``source`` (see :func:`fit`).  An entry of None or ``{}`` starts
        from the values in the XCM file.
    n_procs : int, optional
        Number of fits run at the same time.  Default is 1.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
    save_best_xcm : str or path-like, optional
        If given, the best fit is saved to this XCM file.
    **fit_kwargs
        Further keyword arguments passed to :func:`fit`.

    Returns
    -------
    results : list of FitResult
        Successful fits, sorted by increasing statistic.
    failures : list of tuple
        ``(start, exception)`` for each fit that failed.

    Notes
    -----
    Failed fits are reported rather than raised, since some starting points
    may lie in regions where a model cannot be evaluated.  We recommend
    checking `failures` before relying on the result.

    Examples
    --------
    >>> starts = [None] + [{"perturb_seed": s} for s in range(16)]
    >>> results, failures = multistart("fit.xcm", starts, n_procs=8)  # doctest: +SKIP
    >>> results[0].statistic  # doctest: +SKIP
    """
    specs = [dict(s or {}) for s in starts]

    def one(spec):
        try:
            return fit(xcm, start=spec.get("start"), perturb_seed=spec.get("perturb_seed"),
                       source=spec.get("source", ""), lmod=lmod, **fit_kwargs), None
        except Exception as exc:  # reported, not raised
            return None, (spec, exc)

    with ThreadPoolExecutor(max_workers=max(1, n_procs)) as executor:
        outcomes = list(executor.map(one, specs))
    results = sorted((r for r, _ in outcomes if r is not None), key=lambda r: r.statistic)
    failures = [f for _, f in outcomes if f is not None]
    if save_best_xcm and results:
        best = results[0]
        fit(xcm, start=best.values, lmod=lmod, save_xcm=save_best_xcm, perform=False)
    return results, failures


def polish(xcm, sample_results, lm_result=None, lmod=None, save_xcm=None, **fit_kwargs):
    """Refine the best fit by starting the minimizer from posterior samples.

    Parameters
    ----------
    xcm : str or path-like
        XCM file containing the data and the model.
    sample_results : iterable of SampleResult
        Posteriors of the model, for example from runs with different seeds.
        A fit is started from the highest-likelihood sample of each.
    lm_result : FitResult, optional
        An earlier fit, included among the candidates.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
    save_xcm : str or path-like, optional
        If given, the chosen fit is saved to this XCM file.
    **fit_kwargs
        Further keyword arguments passed to :func:`fit`.

    Returns
    -------
    best : FitResult
        Candidate with the lowest statistic.
    candidates : list of FitResult
        All candidates, including `lm_result`.

    Raises
    ------
    ValueError
        If there are no candidates.

    Notes
    -----
    A minimizer started from a single point can converge to a local minimum.
    The posterior samples cover the whole region of high likelihood, so the
    highest-likelihood sample is often a better starting point.  We
    recommend running this function after sampling and reporting the
    resulting fit as the best fit.
    """
    candidates = [] if lm_result is None else [lm_result]
    for result in sample_results:
        best = result.best_sample()
        candidates.append(fit(xcm, start=best, lmod=lmod, source=f"posterior:{result.label}", **fit_kwargs))
    if not candidates:
        raise ValueError("nothing to polish")
    chosen = min(candidates, key=lambda r: r.statistic)
    if save_xcm:
        fit(xcm, start=chosen.values, lmod=lmod, save_xcm=save_xcm, perform=False)
    return chosen, candidates
