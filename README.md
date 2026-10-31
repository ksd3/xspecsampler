# xspecsampler

`xspecsampler` is a Python package for Bayesian inference with [XSPEC](https://heasarc.gsfc.nasa.gov/xanadu/xspec/) spectral models. It provides robust maximum-likelihood fitting, posterior sampling and evidence estimation for models that are too expensive to explore with standard Markov chain Monte Carlo, such as the relativistic reflection model [relxill](https://www.sternwarte.uni-erlangen.de/~dauser/research/relxill/).

The likelihood is evaluated in parallel by a pool of persistent [PyXspec](https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html) processes ([`XspecLikelihood`](src/xspecsampler/likelihood.py)). Maximum-likelihood fits use XSPEC's [Levenberg–Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) minimizer from multiple starting points ([`fit`, `multistart`](src/xspecsampler/lm.py)); each fit runs in an isolated XSPEC process, is never allowed to end at a higher statistic than its starting point, and is re-evaluated before it is returned. Posterior distributions and Bayesian evidences are computed with [pocoMC](https://github.com/minaskar/pocomc), a preconditioned sequential Monte Carlo sampler ([`sample`](src/xspecsampler/sampler.py)).

Since a sampler may converge to a secondary mode without any indication of failure, we recommend running it through [`sample_with_gates`](src/xspecsampler/gates.py). This function accepts a posterior only if its effective sample size and evidence uncertainty meet given thresholds and its best sample lies close to the maximum-likelihood fit; otherwise the run is repeated with more particles. The final best fit is then obtained with [`polish`](src/xspecsampler/lm.py), which restarts the minimizer from the highest-likelihood posterior sample.

In addition, the [`archive`](src/xspecsampler/archive.py) module searches the [HEASARC](https://heasarc.gsfc.nasa.gov/) archive for observations by [NuSTAR](https://heasarc.gsfc.nasa.gov/docs/nustar/), [NICER](https://heasarc.gsfc.nasa.gov/docs/nicer/), [Swift](https://swift.gsfc.nasa.gov/), [XMM-Newton](https://heasarc.gsfc.nasa.gov/docs/xmm/), [Chandra](https://cxc.harvard.edu/), [Suzaku](https://heasarc.gsfc.nasa.gov/docs/suzaku/) and [RXTE](https://heasarc.gsfc.nasa.gov/docs/xte/), identifies simultaneous observations with different missions, and downloads the data, and the [`phase`](src/xspecsampler/phase.py) module computes orbital phases from a linear ephemeris.

## Installation

The package can be installed directly from GitHub with pip:

```
pip install "xspecsampler[sampler,archive] @ git+https://github.com/<user>/xspecsampler"
```

See the pip documentation on [installing from version control](https://pip.pypa.io/en/stable/topics/vcs-support/) for installing a specific branch or tag.

PyXspec is distributed with [HEASoft](https://heasarc.gsfc.nasa.gov/lheasoft/) and is not available from PyPI, so the package must be used from Python running inside an initialised HEASoft environment. Local models such as relxill or [reflkerr](https://users.camk.edu.pl/mitsza/reflkerr/) have to be built beforehand; their locations are passed through the `lmod` argument, which corresponds to XSPEC's [`lmod`](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSlmod.html) command.

The optional dependencies are defined in [`pyproject.toml`](pyproject.toml). The `sampler` extra installs pocoMC 1.2 and [PyTorch](https://pytorch.org/), and the `archive` extra installs [astropy](https://www.astropy.org/), whose [VOTable reader](https://docs.astropy.org/en/stable/io/votable/index.html) is used to parse the HEASARC responses. The prior and diagnostic functions work without either extra.

## Usage

### Fitting and sampling a model

```python
import json
import xspecsampler as xs

xcm = "fit.xcm"                        # data, model and starting values, as saved by XSPEC
lmod = {"relxill": "/path/to/relxill"}  # overrides lmod paths written in the XCM

info = xs.read_xcm(xcm, lmod=lmod)
print(info.labels)                     # ['DG1:zxipcf.Nh', ..., 'DG2:constant.factor']

# LM fits from the XCM values and from seeded jitters around them
fits, failed = xs.multistart(xcm, [{}] + [{"perturb_seed": s} for s in range(1, 5)],
                             n_procs=8, lmod=lmod)
best = fits[0]

# priors: one rule per thawed parameter, fixed in advance
manifest = json.load(open("priors.json"))
priors = xs.resolve_priors(xs.resolve_profile(manifest, "standard"), info.parameters)

result, history = xs.sample_with_gates(xcm, priors, "out/seed42", seed=42, n_procs=8, lmod=lmod,
                                       reference_statistic=best.statistic)
print(result.log_evidence, result.log_evidence_error, result.ess)

# posterior best fit: LM restarted from the best posterior sample
polished, _ = xs.polish(xcm, [result], lm_result=best, lmod=lmod, save_xcm="out/best_fit.xcm")
```

Parameters are identified by labels of the form `DG<data group>:<component>.<name>`, for example `DG2:constant.factor`. The script [`examples/fit_and_sample.py`](examples/fit_and_sample.py) runs the complete procedure for an arbitrary XCM file, and [`examples/priors_reflection.json`](examples/priors_reflection.json) contains prior profiles for relxill-based reflection models. The format of prior files is documented in [`src/xspecsampler/priors.py`](src/xspecsampler/priors.py). For the sampling method itself, we refer the reader to [Karamanis et al. (2022a)](https://arxiv.org/abs/2207.05652), [Karamanis et al. (2022b)](https://arxiv.org/abs/2207.05660) and the [pocoMC documentation](https://pocomc.readthedocs.io/).

### Finding observations

```python
from xspecsampler import archive, phase

ra, dec = 299.5903, 35.2016                         # or archive.resolve_name("Cyg X-1")
nustar = archive.query("nustar", ra, dec, radius=0.25, columns=("issue_flag",))
nicer = archive.query("nicer", ra, dec, radius=0.25)
for nu, ni, seconds in archive.simultaneous(nustar, nicer, min_overlap=600):
    print(nu.obsid, ni.obsid, round(seconds))

url = archive.archive_directory(nustar[0])          # HEASARC archive directory
archive.download(url, "data/" + nustar[0].obsid, include=["*_cl.evt.gz"])

# orbital phase interval (Brocksopp et al. 1999 ephemeris for Cyg X-1)
phase.phase_interval(nustar[0].start, nustar[0].end, period=5.599829, t0=41874.207)
```

A query can be restricted to a time interval with the `start` and `end` arguments (MJD), and any additional [ADQL](https://www.ivoa.net/documents/ADQL/) condition on the mission's master table can be supplied through `where`; for instance, `where="xrt_expo_wt > 0"` selects Swift/XRT observations with windowed-timing data. The available columns are listed in the HEASARC documentation of the [NuSTAR](https://heasarc.gsfc.nasa.gov/W3Browse/all/numaster.html), [NICER](https://heasarc.gsfc.nasa.gov/W3Browse/all/nicermastr.html), [Swift](https://heasarc.gsfc.nasa.gov/W3Browse/all/swiftmastr.html), [XMM-Newton](https://heasarc.gsfc.nasa.gov/W3Browse/all/xmmmaster.html), [Chandra](https://heasarc.gsfc.nasa.gov/W3Browse/all/chanmaster.html), [Suzaku](https://heasarc.gsfc.nasa.gov/W3Browse/all/suzamaster.html) and [RXTE](https://heasarc.gsfc.nasa.gov/W3Browse/all/xtemaster.html) master tables.

Note that `simultaneous` compares the start and end times of the observations rather than their good time intervals. The result should therefore be regarded as a list of candidates, and we recommend verifying the actual overlap after the data have been reduced.

Queries are sent to the HEASARC [TAP service](https://www.ivoa.net/documents/TAP/) (see the HEASARC [Virtual Observatory pages](https://heasarc.gsfc.nasa.gov/docs/archive/vo/)), and archive locations are obtained from its [DataLink](https://www.ivoa.net/documents/DataLink/) service, which also underlies [Xamin](https://heasarc.gsfc.nasa.gov/xamin/). Target names are resolved with the CDS Sesame service. The ephemeris used in the example is that of [Brocksopp et al. (1999)](https://arxiv.org/abs/astro-ph/9812077).

## Package contents

| Module | Contents |
|---|---|
| [`xcm`](src/xspecsampler/xcm.py) | `read_xcm`: thawed parameters (label, XSPEC index, value, soft and hard limits) and the statistic of an XCM |
| [`likelihood`](src/xspecsampler/likelihood.py) | `XspecLikelihood`: persistent workers, result cache, pocoMC pool interface |
| [`lm`](src/xspecsampler/lm.py) | `fit`, `multistart`, `polish`, `FitResult` |
| [`priors`](src/xspecsampler/priors.py) | prior manifests with `inherits`/`replace`/`add`, `resolve_priors`, `profile_hash` |
| [`sampler`](src/xspecsampler/sampler.py) | `sample`, `SampleResult` (`load`, `load_legacy`, `best_sample`, `resample`) |
| [`gates`](src/xspecsampler/gates.py) | `QualityGates`, `DEFAULT_TIERS`, `sample_with_gates` |
| [`diagnostics`](src/xspecsampler/diagnostics.py) | weighted quantiles, effective sample size, per-parameter summaries, `compare_seeds`, `summed_evidence` |
| [`archive`](src/xspecsampler/archive.py) | `query`, `search`, `resolve_name`, `simultaneous`, `data_links`, `archive_directory`, `download` |
| [`phase`](src/xspecsampler/phase.py) | `orbital_phase`, `phase_interval`, `phase_distance`, `propagated_phase_error` |

The XSPEC worker processes are implemented in [`_workers.py`](src/xspecsampler/_workers.py), and the public interface is defined in [`__init__.py`](src/xspecsampler/__init__.py).

The function `sample` writes three files to its output directory. `posterior_samples.npz` contains the samples, their weights and log-likelihoods, and the parameter labels; `posterior_summary.csv` lists the 5th, 16th, 50th, 84th and 95th percentiles of each parameter; and `status.json` records the evidence, the effective sample size, the best statistic, the sampler settings, the priors and the software versions. A parameter is flagged as railed in the summary if its 5th (or 95th) percentile lies within 2% of the prior width of the corresponding prior bound.

## Notes

PyXspec stores the state of XSPEC in global objects. For this reason every XSPEC operation is carried out in a separate process, and XSPEC is never imported into the calling Python session.

Bayesian evidences are only comparable between runs that use identical priors. We therefore recommend fixing the priors before any fitting, rather than deriving them from a fit result, and storing the value returned by `profile_hash` together with each result, so that it can be verified that two evidences were computed under the same priors.

Priors should not extend beyond the tabulated range of a model. For example, [`zxipcf`](https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelZxipcf.html) is only defined for N_H ≥ 0.05 × 10²² cm⁻²; below this value XSPEC returns the previously computed spectrum instead of an error, so the likelihood is ill-defined.

XSPEC resolves the names of data and response files, including those given in spectrum headers, relative to the working directory. When an XCM file is saved to a directory other than that of its data, the package records the data directory in the first line of the file (`# xspecsampler data directory: ...`) and changes to that directory when the file is restored. XSPEC itself treats this line as a comment, so such a file should be loaded in XSPEC only from within the data directory.

With `device="cuda"`, the normalizing flow of pocoMC is trained on the GPU. This option requires pocoMC 1.2.6, because it replaces two internal functions of that version. By default all computations run on the CPU.

## Testing

```
pytest                          # pure-Python tests; XSPEC tests are skipped without PyXspec
XSPECSAMPLER_REFERENCE_ROOT=/path/to/reference pytest   # inside HEASoft, with RELXILL_PATH set
```

The test suite uses [pytest](https://docs.pytest.org/) and is located in [`tests/`](tests/). The XSPEC tests ([`test_reference_results.py`](tests/test_reference_results.py) and [`test_xspec_sampler.py`](tests/test_xspec_sampler.py)) compare the package with a directory of stored reference results, given by `XSPECSAMPLER_REFERENCE_ROOT`. They verify that the thawed parameters and the statistic of a reference XCM are recovered, that the stored log-likelihoods of posterior samples are reproduced to within 10⁻⁶, that the stored best fit is replayed, and that the stored seed-agreement table is reproduced. They also run the complete procedure of multistart fitting, gated sampling and polishing on a simple model.

The network test in [`test_archive.py`](tests/test_archive.py) queries HEASARC for Cyg X-1 and checks that 27 NuSTAR observations with issue flag 0 are found up to July 2024, and that NICER observed simultaneously with 13 of the 26 that remain after excluding one 0.2 ks pointing. Tests that require neither XSPEC nor a network connection are collected in [`test_core.py`](tests/test_core.py).

## Acknowledgements and licence

The parallel likelihood evaluation is derived from [xspec_emcee](https://github.com/jeremysanders/xspec_emcee) by Jeremy Sanders. `xspecsampler` is distributed under the [MIT licence](https://opensource.org/license/mit); see [`LICENSE`](LICENSE).
