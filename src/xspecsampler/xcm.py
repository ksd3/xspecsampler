"""Reading of XCM files.

XSPEC is never imported into the calling process.  All XSPEC operations run in
worker subprocesses (``python -m xspecsampler._workers``), which return their
results as framed JSON on standard output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ._protocol import RESULT_END, RESULT_START


def worker_environment():
    """Return the environment for worker subprocesses.

    Returns
    -------
    dict
        A copy of ``os.environ`` in which ``PYTHONPATH`` starts with the
        directory containing this package, so that the workers import the
        same version of it.
    """
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def worker_command(command, *extra):
    """Return the command line of a worker subprocess."""
    return [sys.executable, "-u", "-m", "xspecsampler._workers", command, *extra]


def lmod_arguments(lmod):
    """Convert a mapping of local models into ``--lmod NAME=PATH`` arguments."""
    out = []
    for name, path in (lmod or {}).items():
        out += ["--lmod", f"{name}={path}"]
    return out


def run_worker(command, *extra, stdin=None, timeout=None):
    """Run a worker subprocess to completion.

    Parameters
    ----------
    command : {'metadata', 'likelihood', 'fit'}
        Worker command.
    *extra : str
        Additional command-line arguments.
    stdin : str, optional
        Text passed to the worker on standard input.
    timeout : float, optional
        Maximum run time in seconds.

    Returns
    -------
    object
        The decoded JSON result of the worker.

    Raises
    ------
    RuntimeError
        If the worker exits with a non-zero status or returns no result.
    subprocess.TimeoutExpired
        If the worker does not finish within `timeout`.
    """
    proc = subprocess.run(
        worker_command(command, *extra), input=stdin, capture_output=True, text=True,
        env=worker_environment(), timeout=timeout,
    )
    out = proc.stdout
    start, end = out.rfind(RESULT_START), out.rfind(RESULT_END)
    if proc.returncode != 0 or start < 0 or end < start:
        raise RuntimeError(
            f"XSPEC worker '{command}' failed (returncode {proc.returncode}):\n"
            f"{out[-4000:]}\n{proc.stderr[-2000:]}"
        )
    return json.loads(out[start + len(RESULT_START):end])


@dataclass(frozen=True)
class Parameter:
    """Description of a thawed XSPEC parameter.

    Attributes
    ----------
    label : str
        Parameter label, ``DG<group>:<component>.<name>``.
    data_group : int
        Data group to which the parameter belongs.
    component, name : str
        Model component and parameter name.
    xspec_index : int
        Global XSPEC parameter number, as used by the ``newpar`` command.  The
        parameters of data groups 2, 3, ... are numbered after those of
        group 1.
    local_index : int
        Parameter number within the model of its data group.
    value, delta : float
        Current value and fit step size.
    hard_min, soft_min, soft_max, hard_max : float
        XSPEC parameter limits.
    unit : str
        Parameter unit.
    """
    label: str
    data_group: int
    component: str
    name: str
    xspec_index: int
    local_index: int
    value: float
    delta: float
    hard_min: float
    soft_min: float
    soft_max: float
    hard_max: float
    unit: str = ""

    def as_dict(self):
        """Return the attributes as a dictionary."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class XcmInfo:
    """Thawed parameters and fit statistic of an XCM file.

    Attributes
    ----------
    xcm : pathlib.Path
        Absolute path of the XCM file.
    parameters : tuple of Parameter
        Thawed parameters, in XSPEC order.
    statistic : float
        Fit statistic of the restored model.
    dof : int
        Degrees of freedom.
    """
    xcm: Path
    parameters: tuple
    statistic: float
    dof: int

    @property
    def labels(self):
        """list of str: Labels of the thawed parameters."""
        return [p.label for p in self.parameters]


def read_xcm(xcm, lmod=None, timeout=600):
    """Read the thawed parameters and fit statistic of an XCM file.

    The file is restored in a separate XSPEC process, so XSPEC is not
    imported into the calling session.

    Parameters
    ----------
    xcm : str or path-like
        XCM file containing the data, the model and the parameter values.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
        These locations replace any ``lmod`` paths written in the file.
    timeout : float, optional
        Maximum run time of the XSPEC process in seconds.  Default is 600.

    Returns
    -------
    XcmInfo
        Thawed parameters, fit statistic and degrees of freedom.

    Raises
    ------
    RuntimeError
        If XSPEC fails to restore the file.

    Examples
    --------
    >>> info = read_xcm("fit.xcm", lmod={"relxill": "/opt/relxill"})  # doctest: +SKIP
    >>> info.labels[:2]  # doctest: +SKIP
    ['DG1:zxipcf.Nh', 'DG1:zxipcf.log_xi']
    """
    xcm = Path(xcm).resolve()
    result = run_worker("metadata", "--xcm", str(xcm), *lmod_arguments(lmod), timeout=timeout)
    return XcmInfo(
        xcm=xcm,
        parameters=tuple(Parameter(**row) for row in result["parameters"]),
        statistic=float(result["statistic"]),
        dof=int(result["dof"]),
    )
