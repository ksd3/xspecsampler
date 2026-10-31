# Notices

## xspec_emcee

The parallel likelihood evaluation in `src/xspecsampler/likelihood.py` and the likelihood loop in `src/xspecsampler/_workers.py` are derived from [xspec_emcee](https://github.com/jeremysanders/xspec_emcee) by Jeremy Sanders, in particular its persistent-worker pool and its parameter (`%e`) and statistic (10 significant digits) conventions.
