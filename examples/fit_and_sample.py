"""Fit and sample one XSPEC model with xspecsampler.

LM multistart (XCM values plus seeded jitters), quality-gated pocoMC
posterior with the default four sampler tiers, then the posterior-informed
polish.  For a relxill model one posterior takes hours on 8 workers; pass
--quick for a short run that only checks the workflow.

Usage (inside a HEASoft environment):

    python fit_and_sample.py fit.xcm --lmod relxill=$RELXILL_PATH \\
        --priors priors_reflection.json --profile standard \\
        --out run --n-procs 8 [--quick] [--device cuda]
"""

import argparse
import json
from pathlib import Path

import xspecsampler as xs

QUICK = [("quick", {"n_effective": 128, "n_active": 64, "n_total": 512, "n_evidence": 512,
                    "train_frequency": 1, "flow_epochs": 100, "flow_patience": 10, "save_every": 2})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xcm", type=Path, help="XCM with data, model and starting values")
    parser.add_argument("--lmod", action="append", default=[], metavar="NAME=PATH",
                        help="local model package and its build directory (repeatable)")
    parser.add_argument("--priors", type=Path, default=Path(__file__).with_name("priors_reflection.json"))
    parser.add_argument("--profile", default="standard", help="prior profile in --priors")
    parser.add_argument("--out", type=Path, default=Path("run"))
    parser.add_argument("--n-procs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    xcm = args.xcm
    lmod = dict(item.split("=", 1) for item in args.lmod)
    info = xs.read_xcm(xcm, lmod=lmod)
    print(f"{len(info.parameters)} thawed parameters; statistic of the XCM as saved: {info.statistic:.2f}/{info.dof}")

    starts = [{"source": "xcm"}] + [{"perturb_seed": s, "source": f"jitter{s}"} for s in range(1, 5)]
    fits, failed = xs.multistart(xcm, starts, n_procs=args.n_procs, lmod=lmod)
    best = fits[0]
    print(f"LM: best {best.statistic:.2f} from '{best.source}' ({len(failed)} starts failed)")

    manifest = json.loads(args.priors.read_text())
    priors = xs.resolve_priors(xs.resolve_profile(manifest, args.profile), info.parameters)
    result, history = xs.sample_with_gates(
        xcm, priors, args.out / f"seed{args.seed}", seed=args.seed,
        tiers=QUICK if args.quick else None, gates=xs.QualityGates(min_ess=100) if args.quick else xs.QualityGates(),
        reference_statistic=best.statistic, n_procs=args.n_procs, lmod=lmod, device=args.device,
    )
    for tier in history:
        print(f"  tier {tier['tier']}: ESS {tier['ess']:.0f}, lnZ {tier['log_evidence']:.2f} "
              f"+/- {tier['log_evidence_error']:.2f}, failures: {tier['failures'] or 'none'}")

    polished, _ = xs.polish(xcm, [result], lm_result=best, lmod=lmod, save_xcm=args.out / "best_fit.xcm")
    print(f"posterior best fit: {polished.statistic:.2f}/{polished.dof} (source {polished.source or 'LM'})")
    for row in result.summary:
        print(f"  {row['label']:32s} {row['p50']:.4g} (+{row['error_upper']:.2g} -{row['error_lower']:.2g})"
              + ("  railed" if row["railed_lower"] or row["railed_upper"] else ""))


if __name__ == "__main__":
    main()
