"""Entry points of the XSPEC worker subprocesses.

PyXspec keeps the state of XSPEC in process-global objects.  Every operation
that uses XSPEC therefore runs in its own subprocess, started with
``python -m xspecsampler._workers <command>``.  The commands are:

metadata
    Restore an XCM file and print the table of thawed parameters as JSON.
likelihood
    Read comma-separated parameter vectors from standard input and print
    the fit statistic of each.  The process runs until it receives ``QUIT``.
fit
    Read a JSON job from standard input, fit the model from the requested
    starting point, and print the result as JSON.

The likelihood loop is derived from the PyXspec backend of xspec_emcee
(J. Sanders), used with the author's permission.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

from xspecsampler._protocol import READY, RESULT_END, RESULT_START


# --------------------------------------------------------------------------- session

def parse_lmod(items):
    """Convert ``NAME=PATH`` arguments into a dictionary."""
    out = {}
    for item in items or []:
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"--lmod expects NAME=PATH, got {item!r}")
        out[name] = path
    return out


DATA_DIRECTORY_TAG = "# xspecsampler data directory:"


def data_directory(xcm):
    """Return the directory against which the file names in an XCM file are resolved.

    This is the directory recorded in the file by :func:`_save_xcm`, if any,
    and otherwise the directory of the file itself.
    """
    xcm = Path(xcm).resolve()
    for line in xcm.read_text().splitlines()[:5]:
        if line.startswith(DATA_DIRECTORY_TAG):
            return Path(line[len(DATA_DIRECTORY_TAG):].strip())
    return xcm.parent


def restore(xcm, lmod=None):
    """Restore an XCM file.

    Parameters
    ----------
    xcm : str or path-like
        XCM file.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
        An ``lmod`` line of the file whose package is in `lmod` is replaced
        by the given directory; other lines are kept.  This allows files
        written on another machine to be restored.

    Notes
    -----
    File names in the XCM file are resolved against the directory returned
    by :func:`data_directory`.
    """
    from xspec import AllModels, Fit, Plot, Xset

    xcm = Path(xcm).resolve()
    lmod = dict(lmod or {})
    os.chdir(data_directory(xcm))
    Xset.chatter = 0
    Xset.logChatter = 0
    Plot.device = "/null"
    Fit.query = "no"
    kept = []
    for line in xcm.read_text().splitlines():
        match = re.match(r"\s*lmod\s+(\S+)", line)
        if match and match.group(1) in lmod:
            continue
        kept.append(line)
    for name, path in lmod.items():
        AllModels.lmod(name, str(path))
    with tempfile.NamedTemporaryFile("w", suffix=".xcm", delete=False) as handle:
        handle.write("\n".join(kept) + "\n")
        temporary = handle.name
    try:
        Xset.restore(temporary)
    finally:
        os.unlink(temporary)


def thawed_parameters():
    """Return the thawed parameters of all data groups, in XSPEC order.

    Parameters that are frozen or linked to another parameter are excluded.
    The ``xspec_index`` of each parameter is the global index used by the
    XSPEC ``newpar`` command, in which the parameters of data groups 2, 3, ...
    are numbered after those of group 1.
    """
    from xspec import AllData, AllModels

    rows = []
    for group in range(1, AllData.nGroups + 1):
        model = AllModels(group)
        for component_name in model.componentNames:
            component = getattr(model, component_name)
            for parameter_name in component.parameterNames:
                parameter = getattr(component, parameter_name)
                if parameter.frozen or parameter.link:
                    continue
                values = parameter.values
                if len(values) < 6:  # switch parameters have a single value
                    continue
                rows.append({
                    "label": f"DG{group}:{component_name}.{parameter_name}",
                    "data_group": group,
                    "component": component_name,
                    "name": parameter_name,
                    "xspec_index": int(model.startParIndex) + int(parameter.index) - 1,
                    "local_index": int(parameter.index),
                    "value": float(values[0]),
                    "delta": float(values[1]),
                    "hard_min": float(values[2]),
                    "soft_min": float(values[3]),
                    "soft_max": float(values[4]),
                    "hard_max": float(values[5]),
                    "unit": getattr(parameter, "unit", "") or "",
                })
    return rows


def set_values(rows, values):
    """Set the values of the thawed parameters with a single model recalculation."""
    from xspec import AllModels

    updates = {}
    for row, value in zip(rows, values):
        updates.setdefault(int(row["data_group"]), {})[int(row["local_index"])] = float(value)
    arguments = []
    for group in sorted(updates):
        arguments.extend((AllModels(group), updates[group]))
    AllModels.setPars(*arguments)


def current_values(rows):
    from xspec import AllModels

    return [float(AllModels(int(r["data_group"]))(int(r["local_index"])).values[0]) for r in rows]


def statistic():
    from xspec import Fit

    return float(Fit.statistic)


# --------------------------------------------------------------------------- commands

def cmd_metadata(args):
    restore(args.xcm, parse_lmod(args.lmod))
    from xspec import Fit

    rows = thawed_parameters()
    print(RESULT_START + json.dumps({
        "parameters": rows,
        "statistic": statistic(),
        "dof": int(Fit.dof),
    }) + RESULT_END, flush=True)
    return 0


def cmd_likelihood(args):
    restore(args.xcm, parse_lmod(args.lmod))
    rows = thawed_parameters()
    labels = [r["label"] for r in rows]
    if args.labels and args.labels.split(",") != labels:
        raise RuntimeError(f"worker parameter order {labels} differs from the requested {args.labels}")
    print(READY, flush=True)
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "QUIT":
            return 0
        values = line.split(",")
        if len(values) != len(rows):
            raise RuntimeError(f"received {len(values)} values for {len(rows)} parameters")
        set_values(rows, [float(v) for v in values])
        # Ten significant digits, as XSPEC's Tcl `tcloutr stat` reports it.
        print(f"{RESULT_START}{statistic():.10g}{RESULT_END}", flush=True)
    return 0


def _perturb(rows, rng, scale):
    """Perturb the current parameter values with Gaussian noise.

    The standard deviation for each parameter is
    ``max(5 * |delta|, scale * (soft_max - soft_min))``.  The perturbed values
    are clipped to the soft limits.
    """
    values = current_values(rows)
    out = []
    for row, value in zip(rows, values):
        lo, hi = row["soft_min"], row["soft_max"]
        width = hi - lo
        if width <= 0:
            out.append(value)
            continue
        sigma = max(5.0 * abs(row["delta"]), scale * width)
        out.append(float(min(max(rng.normal(value, sigma), lo + 1e-12 * width), hi - 1e-12 * width)))
    set_values(rows, out)


def _save_xcm(path, original_xcm):
    """Save the current state of XSPEC to an XCM file.

    ``Xset.save`` writes neither the ``lmod`` lines nor the ``energies`` line.
    These are copied from the original file, and the ``energies`` line is
    placed before the model definition.
    """
    from xspec import Xset

    path = Path(path).resolve()
    body_path = path.with_suffix(path.suffix + ".body")
    if body_path.exists():
        body_path.unlink()
    Xset.save(str(body_path), info="a")
    body = body_path.read_text()
    body_path.unlink()
    original_path = Path(original_xcm).resolve()
    original = original_path.read_text().splitlines()
    lmods = [line for line in original if re.match(r"\s*lmod\s", line)]
    header = ""
    if path.parent != original_path.parent:
        # XSPEC resolves spectrum, background and response names (including
        # those in spectrum headers) against the working directory, so record
        # where the data are; restore() changes to that directory.
        header = f"{DATA_DIRECTORY_TAG} {data_directory(original_path)}\n"
    energies = [line for line in original if re.match(r"\s*energies\s", line)]
    if energies and not re.search(r"(?m)^\s*energies\s", body):
        body = re.sub(r"(?m)^(model\s+)", "\n".join(energies) + "\n\\1", body, count=1)
    path.write_text(header + "".join(f"{line}\n" for line in lmods) + body)


def cmd_fit(args):
    import numpy as np

    job = json.loads(sys.stdin.read())
    restore(job["xcm"], job.get("lmod"))
    from xspec import Fit

    Fit.nIterations = int(job.get("n_iterations", 1000))
    rows = thawed_parameters()
    labels = [r["label"] for r in rows]
    start = job.get("start") or {}
    unknown = sorted(set(start) - set(labels))
    if unknown and job.get("strict_start", True):
        raise RuntimeError(f"start values for parameters that are not thawed: {unknown}")
    if start:
        values = current_values(rows)
        for i, row in enumerate(rows):
            if row["label"] in start:
                values[i] = float(np.clip(float(start[row["label"]]), row["hard_min"], row["hard_max"]))
        set_values(rows, values)
    if job.get("perturb_seed") is not None:
        _perturb(rows, np.random.default_rng(int(job["perturb_seed"])), float(job.get("perturb_scale", 0.05)))

    start_values = current_values(rows)
    start_stat = statistic()
    if job.get("fit", True):
        Fit.perform()
    fitted = statistic()
    # Keep the better of the start and the fit, so a fit that wanders uphill
    # cannot make the result worse than its starting point.
    if np.isfinite(fitted) and fitted <= start_stat:
        best, best_values = fitted, current_values(rows)
    else:
        best, best_values = start_stat, start_values
    set_values(rows, best_values)
    replay = statistic()
    tolerance = 5e-5 * max(1.0, abs(best))
    if abs(replay - best) > tolerance:
        raise RuntimeError(f"best-fit replay mismatch: {replay} versus {best}")
    if job.get("save_xcm"):
        _save_xcm(job["save_xcm"], job["xcm"])
    print(RESULT_START + json.dumps({
        "labels": labels,
        "start_statistic": start_stat,
        "fit_statistic": fitted,
        "statistic": best,
        "dof": int(Fit.dof),
        "values": dict(zip(labels, best_values)),
        "start_values": dict(zip(labels, start_values)),
        "ignored_start_labels": unknown,
    }) + RESULT_END, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m xspecsampler._workers")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("metadata", "likelihood"):
        p = sub.add_parser(name)
        p.add_argument("--xcm", required=True)
        p.add_argument("--lmod", action="append", default=[])
        if name == "likelihood":
            p.add_argument("--labels", default="")
    sub.add_parser("fit")
    args = parser.parse_args(argv)
    return {"metadata": cmd_metadata, "likelihood": cmd_likelihood, "fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        code = 1
    raise SystemExit(code)
