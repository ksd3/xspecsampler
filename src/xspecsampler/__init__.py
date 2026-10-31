"""Bayesian inference for XSPEC spectral models.

``xspecsampler`` fits XSPEC models, samples their posteriors with pocoMC and
estimates their Bayesian evidences.  PyXspec (part of HEASoft) is required for
any operation that evaluates a model, and pocoMC and PyTorch (the ``sampler``
extra) are required for sampling.  Neither is imported until it is needed.

Examples
--------
::

    import xspecsampler as xs

    info = xs.read_xcm("fit.xcm")
    best, failed = xs.multistart("fit.xcm", starts, n_procs=8)
    priors = xs.resolve_priors(xs.resolve_profile(manifest, "standard"), info.parameters)
    result, history = xs.sample_with_gates("fit.xcm", priors, "out/seed42", seed=42,
                                           n_procs=8, reference_statistic=best[0].statistic)
    polished, _ = xs.polish("fit.xcm", [result], lm_result=best[0])
"""

__version__ = "0.1.0.dev0"

from .diagnostics import compare_seeds, effective_sample_size, parameter_summary, summed_evidence, weighted_quantiles
from .priors import Prior, profile_hash, resolve_priors, resolve_profile
from .xcm import Parameter, XcmInfo, read_xcm

_LAZY = {
    "XspecLikelihood": ("likelihood", "XspecLikelihood"),
    "FitResult": ("lm", "FitResult"),
    "fit": ("lm", "fit"),
    "multistart": ("lm", "multistart"),
    "polish": ("lm", "polish"),
    "SampleResult": ("sampler", "SampleResult"),
    "sample": ("sampler", "sample"),
    "QualityGates": ("gates", "QualityGates"),
    "QualityGateError": ("gates", "QualityGateError"),
    "DEFAULT_TIERS": ("gates", "DEFAULT_TIERS"),
    "sample_with_gates": ("gates", "sample_with_gates"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module 'xspecsampler' has no attribute {name!r}")


__all__ = [
    "Parameter", "XcmInfo", "read_xcm", "Prior", "resolve_profile", "resolve_priors", "profile_hash",
    "weighted_quantiles", "effective_sample_size", "parameter_summary", "compare_seeds", "summed_evidence",
    *_LAZY,
]
