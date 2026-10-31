"""Posterior sampling and evidence estimation with pocoMC.

pocoMC (preconditioned sequential Monte Carlo; Karamanis et al. 2022) samples
in coordinates in which every prior is uniform, i.e. the log10 of the
parameter for log-uniform priors.  The likelihood is evaluated by the workers
of an :class:`XspecLikelihood`.  A run writes a checkpoint every
``save_every`` iterations, keeps the two most recent checkpoints, and resumes
from the most recent one if it is restarted after an interruption.

A completed run writes three files to its output directory:
``posterior_samples.npz``, ``posterior_summary.csv`` and ``status.json``.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._io import config_hash, load_json, write_json_atomic
from .diagnostics import effective_sample_size, parameter_summary
from .likelihood import XspecLikelihood

# Sampler settings of the first tier of gates.DEFAULT_TIERS.
DEFAULT_SETTINGS = {
    "n_effective": 512, "n_active": 256, "n_total": 2048, "n_evidence": 2048,
    "train_frequency": 1, "flow_epochs": 1000, "flow_patience": 10, "save_every": 5,
}

_STATE = re.compile(r"_(?P<step>[0-9]+|final)\.state$")


def checkpoint_key(path):
    """Return a sort key that orders pocoMC checkpoints by iteration.

    Parameters
    ----------
    path : str or path-like
        Checkpoint file.

    Returns
    -------
    tuple
        Sort key.  The final checkpoint sorts last.

    Notes
    -----
    Checkpoints are ordered by the iteration number in their file name
    because modification times are not reliable on all file systems.
    """
    name = Path(path).name
    match = _STATE.search(name)
    if match is None:
        return (0, -1, name)
    if match.group("step") == "final":
        return (2, 0, name)
    return (1, int(match.group("step")), name)


@dataclass
class SampleResult:
    """Weighted posterior sample and evidence of a model.

    Attributes
    ----------
    label : str
        Name of the run.
    labels : list of str
        Parameter labels, in the column order of `samples`.
    samples : numpy.ndarray, shape (n_samples, n_parameters)
        Posterior samples in physical units.
    weights : numpy.ndarray, shape (n_samples,)
        Normalized importance weights.
    log_likelihood : numpy.ndarray, shape (n_samples,)
        Log-likelihood of each sample.
    log_evidence, log_evidence_error : float
        Natural logarithm of the evidence and its uncertainty.
    ess : float
        Effective sample size of the weights.
    summary : list of dict
        Per-parameter summary from :func:`parameter_summary`.
    status : dict
        Contents of ``status.json``.
    """
    label: str
    labels: list
    samples: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    log_evidence: float
    log_evidence_error: float
    ess: float
    summary: list
    status: dict = field(default_factory=dict)

    @property
    def best_statistic(self):
        """float: Lowest fit statistic among the samples."""
        return float(-2.0 * np.max(self.log_likelihood))

    def best_sample(self):
        """Return the sample with the highest likelihood.

        Returns
        -------
        dict
            Parameter values keyed by label.
        """
        i = int(np.argmax(self.log_likelihood))
        return dict(zip(self.labels, (float(v) for v in self.samples[i])))

    def resample(self, n, seed=0):
        """Draw equally weighted samples from the posterior.

        Parameters
        ----------
        n : int
            Number of draws.
        seed : int, optional
            Random seed.  Default is 0.

        Returns
        -------
        numpy.ndarray, shape (n, n_parameters)
            Samples drawn with replacement according to the weights.  These
            are suitable for corner plots and other plots that expect
            unweighted samples.
        """
        rng = np.random.default_rng(seed)
        return self.samples[rng.choice(len(self.weights), size=n, p=self.weights / self.weights.sum())]

    @classmethod
    def load(cls, directory):
        """Read the output of :func:`sample`.

        Parameters
        ----------
        directory : str or path-like
            Output directory of the run.

        Returns
        -------
        SampleResult
            The stored result.
        """
        directory = Path(directory)
        z = np.load(directory / "posterior_samples.npz")
        status = load_json(directory / "status.json")
        with open(directory / "posterior_summary.csv", newline="", encoding="utf-8") as handle:
            summary = [
                {k: (v if k in ("label", "prior_distribution") else float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)
            ]
        return cls(
            label=status.get("label", directory.name), labels=list(z["parameter_labels"].astype(str)),
            samples=z["samples"], weights=z["weights"], log_likelihood=z["log_likelihood"],
            log_evidence=float(status["log_evidence"]), log_evidence_error=float(status["log_evidence_error"]),
            ess=float(status["posterior_ess"]), summary=summary, status=status,
        )


    @classmethod
    def load_legacy(cls, directory):
        """Read an output directory written in the older format.

        Parameters
        ----------
        directory : str or path-like
            Output directory of the run.

        Returns
        -------
        SampleResult
            The stored result.

        Notes
        -----
        In the older format, ``status.json`` has schema version 2 and the
        rows of ``posterior_summary.csv`` are identified by the columns
        ``data_group``, ``component`` and ``name`` instead of ``label``.
        """
        directory = Path(directory)
        z = np.load(directory / "posterior_samples.npz")
        status = load_json(directory / "status.json")
        summary = []
        with open(directory / "posterior_summary.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out = {"label": f"DG{row['data_group']}:{row['component']}.{row['name']}"}
                out.update({k: float(v) for k, v in row.items()
                            if k not in ("data_group", "component", "name", "prior_distribution")})
                summary.append(out)
        return cls(
            label=status.get("sample_id", directory.name), labels=list(z["parameter_labels"].astype(str)),
            samples=z["samples"], weights=z["weights"], log_likelihood=z["log_likelihood"],
            log_evidence=float(status["log_evidence"]), log_evidence_error=float(status["log_evidence_error"]),
            ess=float(status["posterior_ess"]), summary=summary, status=status,
        )


def _cuda_bridge():
    """Train the pocoMC normalizing flow on a CUDA device.

    Returns
    -------
    device : torch.device
        The CUDA device.
    info : dict
        Name, compute capability and CUDA runtime version of the device.

    Raises
    ------
    RuntimeError
        If the installed pocoMC version is not 1.2.6, or if no CUDA device
        is available.

    Notes
    -----
    pocoMC 1.2.6 converts arrays with module-level helper functions that
    always create CPU tensors.  These functions are replaced here, so that the
    flow is trained on the GPU.  The replacement is specific to that version
    of pocoMC.
    """
    import pocomc
    import pocomc.sampler as sampler_module
    import pocomc.tools as tools
    import torch

    if pocomc.__version__ != "1.2.6":
        raise RuntimeError(f"device='cuda' needs pocoMC 1.2.6, found {pocomc.__version__}; use device='cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is false")
    device = torch.device("cuda:0")

    def to_torch(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def to_numpy(value):
        return value.detach().cpu().numpy()

    tools.numpy_to_torch = to_torch
    tools.torch_to_numpy = to_numpy
    sampler_module.numpy_to_torch = to_torch
    sampler_module.torch_to_numpy = to_numpy
    torch.cuda.set_device(device)
    return device, {
        "name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda_runtime": torch.version.cuda,
    }


def _sampler_class(device):
    import pocomc

    class RotatingCheckpointSampler(pocomc.Sampler):
        """pocoMC sampler that keeps only the two most recent checkpoints.

        On resuming, the requested training settings are restored, since
        pocoMC would otherwise take them from the checkpoint.
        """

        def load_state(self, path):
            super().load_state(path)
            if device is not None:
                self.flow.flow.to(device)
            for attr, key in (("requested_flow_epochs", "epochs"), ("requested_flow_patience", "patience")):
                if hasattr(self, attr):
                    self.train_config[key] = getattr(self, attr)
            if hasattr(self, "requested_train_frequency"):
                self.train_frequency = self.requested_train_frequency

        def save_state(self, path):
            super().save_state(path)
            states = sorted(Path(self.output_dir).glob(f"{self.output_label}_*.state"), key=checkpoint_key)
            for old in states[:-2]:
                old.unlink()

    return RotatingCheckpointSampler


def sample(model, priors, out_dir, seed=42, settings=None, n_procs=1, lmod=None, device="cpu",
           reference_statistic=None, label=None, keep_checkpoints=False, progress=False):
    """Sample the posterior of a model and estimate its evidence.

    Parameters
    ----------
    model : str, path-like or XspecLikelihood
        XCM file, or an open likelihood.  If a file is given, the workers are
        started and stopped by this function.
    priors : list of Prior
        One prior per thawed parameter, in the order of the XCM file.
    out_dir : str or path-like
        Output directory.  It is created if it does not exist.
    seed : int, optional
        Random seed of the sampler.  Default is 42.
    settings : dict, optional
        Sampler settings that replace those in :data:`DEFAULT_SETTINGS`.
    n_procs : int, optional
        Number of likelihood workers, if `model` is a file.  Default is 1.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
    device : {'cpu', 'cuda'}, optional
        Device on which the normalizing flow is trained.  ``'cuda'``
        requires pocoMC 1.2.6.  Default is ``'cpu'``.
    reference_statistic : float, optional
        Statistic of a reference fit, normally the best fit from
        :func:`fit` or :func:`multistart`.  It is stored in ``status.json``
        and used by the quality gates.
    label : str, optional
        Name of the run.  Default is ``<xcm stem>_seed<seed>``.
    keep_checkpoints : bool, optional
        If False (default), the checkpoints are deleted once the run is
        complete.
    progress : bool, optional
        If True, show the pocoMC progress bar.  Default is False.

    Returns
    -------
    SampleResult
        Posterior samples, evidence and summary.

    Raises
    ------
    ValueError
        If the priors are not in the order of the thawed parameters, or if
        `device` is not recognized.
    RuntimeError
        If ``device='cuda'`` cannot be used.

    Notes
    -----
    Calling the function again with the same `out_dir`, `seed` and
    `settings` resumes an interrupted run from its last checkpoint.

    We recommend running at least two seeds per model and comparing them with
    :func:`compare_seeds`, and using :func:`sample_with_gates` rather than
    this function when the required particle numbers are not known in
    advance.

    Examples
    --------
    >>> info = read_xcm("fit.xcm")  # doctest: +SKIP
    >>> priors = resolve_priors(resolve_profile(manifest, "standard"), info.parameters)  # doctest: +SKIP
    >>> result = sample("fit.xcm", priors, "run_seed42", seed=42, n_procs=16)  # doctest: +SKIP
    >>> result.log_evidence, result.ess  # doctest: +SKIP
    """
    import pocomc
    import torch
    from scipy.stats import uniform

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    own = not isinstance(model, XspecLikelihood)
    likelihood = XspecLikelihood(model, n_procs=n_procs, lmod=lmod) if own else model
    label = label or f"{Path(likelihood.info.xcm).stem}_seed{seed}"
    try:
        if [p.label for p in priors] != likelihood.labels:
            raise ValueError(f"prior order {[p.label for p in priors]} differs from the XCM's thawed parameters {likelihood.labels}")
        torch_device, cuda_info = (None, None)
        if device == "cuda":
            torch_device, cuda_info = _cuda_bridge()
        elif device != "cpu":
            raise ValueError("device must be 'cpu' or 'cuda'")
        is_log = np.array([p.is_log for p in priors])
        prior = pocomc.Prior([uniform(loc=a, scale=b - a) for a, b in (p.coordinate_bounds() for p in priors)])

        class Pool:  # pocoMC hands every batch of points to pool.map
            def map(self, _function, points):
                x = np.atleast_2d(np.asarray(list(points), dtype=float))
                x[:, is_log] = np.power(10.0, x[:, is_log])
                return likelihood.map(None, list(x))

        def unused(_):
            raise RuntimeError("pocoMC bypassed the XSPEC pool")

        run_id = hashlib.sha256(f"{label}|{config_hash(settings)}".encode()).hexdigest()[:16]
        Sampler = _sampler_class(torch_device)
        sampler = Sampler(
            prior, unused,
            n_effective=int(settings["n_effective"]), n_active=int(settings["n_active"]),
            pool=Pool(), pytorch_threads=int(settings.get("pytorch_threads", max(1, min(4, n_procs)))),
            train_frequency=int(settings["train_frequency"]),
            train_config={"epochs": int(settings["flow_epochs"]), "patience": int(settings["flow_patience"])},
            random_state=int(seed), output_dir=str(out_dir), output_label=f"poco_{run_id}",
        )
        if torch_device is not None:
            sampler.flow.flow.to(torch_device)
        sampler.requested_train_frequency = int(settings["train_frequency"])
        sampler.requested_flow_epochs = int(settings["flow_epochs"])
        sampler.requested_flow_patience = int(settings["flow_patience"])
        states = sorted(out_dir.glob(f"poco_{run_id}_*.state"), key=checkpoint_key)
        resume = str(states[-1]) if states else None
        started = time.time()
        sampler.run(n_total=int(settings["n_total"]), n_evidence=int(settings["n_evidence"]),
                    progress=progress, save_every=int(settings["save_every"]), resume_state_path=resume)
        wall = time.time() - started
        samples, weights, log_l, log_prior_coord = sampler.posterior(trim_importance_weights=False)
        samples = np.asarray(samples, dtype=float)
        samples[:, is_log] = np.power(10.0, samples[:, is_log])
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        log_l = np.asarray(log_l, dtype=float)
        logz, logz_err = (float(v) for v in sampler.evidence())
        calls = int(sampler.calls)
    finally:
        if own:
            likelihood.close()

    summary = parameter_summary(samples, weights, priors)
    best = float(-2.0 * np.max(log_l))
    status = {
        "schema": "xspecsampler.sample/1",
        "label": label,
        "xcm": str(likelihood.info.xcm),
        "seed": int(seed),
        "settings": settings,
        "priors": [p.__dict__ for p in priors],
        "sampler_complete": True,
        "device": device,
        "cuda_device": cuda_info,
        "n_dimensions": len(priors),
        "n_posterior_samples": int(len(samples)),
        "posterior_ess": effective_sample_size(weights),
        "log_evidence": logz,
        "log_evidence_error": logz_err,
        "posterior_best_statistic": best,
        "reference_statistic": reference_statistic,
        "posterior_best_delta_from_reference": None if reference_statistic is None else best - float(reference_statistic),
        "dof": likelihood.info.dof,
        "n_railed_lower": sum(r["railed_lower"] for r in summary),
        "n_railed_upper": sum(r["railed_upper"] for r in summary),
        "likelihood_calls": calls,
        "xspec_evaluations": likelihood.evaluations,
        "resumed_from": Path(resume).name if resume else None,
        "wall_seconds": wall,
        "pocomc_version": pocomc.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "posterior_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    np.savez_compressed(
        out_dir / "posterior_samples.npz", samples=samples, weights=weights, log_likelihood=log_l,
        log_prior_sample_coordinate=np.asarray(log_prior_coord, dtype=float),
        parameter_labels=np.asarray([p.label for p in priors]),
    )
    write_json_atomic(out_dir / "status.json", status)
    if not keep_checkpoints:
        for path in list(out_dir.glob(f"poco_{run_id}_*.state")) + list(out_dir.glob(f"poco_{run_id}_*.temp")):
            path.unlink()
    return SampleResult(
        label=label, labels=[p.label for p in priors], samples=samples, weights=weights,
        log_likelihood=log_l, log_evidence=logz, log_evidence_error=logz_err,
        ess=status["posterior_ess"], summary=summary, status=status,
    )

