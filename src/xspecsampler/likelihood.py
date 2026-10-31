"""Parallel evaluation of XSPEC likelihoods.

Each worker is a separate process that holds one restored copy of the XCM
file, so that the process-global state of XSPEC is never shared between
evaluations.  Parameter vectors are sent to the workers in ``%e`` format (the
rounding used by xspec_emcee), and the statistic is returned with ten
significant digits.  A given vector therefore always yields the same
likelihood.  Points are dispatched one at a time for load balancing, and
repeated vectors are answered from a cache.

Derived from xspec_emcee and its PyXspec backend (J. Sanders), used with the
author's permission.
"""

from __future__ import annotations

import os
import select
import subprocess
from collections import OrderedDict, deque

import numpy as np

from ._protocol import READY, RESULT_END, RESULT_START
from .xcm import lmod_arguments, read_xcm, worker_command, worker_environment


class _Worker:
    """Persistent likelihood subprocess."""

    def __init__(self, xcm, labels, lmod):
        self.popen = subprocess.Popen(
            worker_command("likelihood", "--xcm", str(xcm), "--labels", ",".join(labels), *lmod_arguments(lmod)),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=worker_environment(),
        )
        self.buffer = ""
        while READY not in self.buffer:
            self._read()
        self.buffer = self.buffer[self.buffer.index(READY) + len(READY):]

    def fileno(self):
        return self.popen.stdout.fileno()

    def _read(self):
        chunk = os.read(self.fileno(), 65536)
        if not chunk:
            raise RuntimeError(
                f"likelihood worker exited (returncode={self.popen.poll()!r}); "
                f"last output: {self.buffer[-2000:]!r}"
            )
        self.buffer += chunk.decode("utf-8", errors="replace")

    def send(self, row):
        self.popen.stdin.write(row + "\n")
        self.popen.stdin.flush()

    def results(self):
        self._read()
        out = []
        while True:
            start = self.buffer.find(RESULT_START)
            end = self.buffer.find(RESULT_END, start + 1) if start >= 0 else -1
            if start < 0 or end < 0:
                break
            out.append(float(self.buffer[start + len(RESULT_START):end]))
            self.buffer = self.buffer[end + len(RESULT_END):]
        return out

    def close(self):
        if self.popen is None:
            return
        if self.popen.poll() is None:
            try:
                self.popen.stdin.write("QUIT\n")
                self.popen.stdin.flush()
                self.popen.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        try:
            self.popen.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            self.popen.wait(timeout=10)
        self.popen = None


def encode(values):
    """Encode a parameter vector as text.

    Parameters
    ----------
    values : sequence of float
        Parameter values.

    Returns
    -------
    str
        Comma-separated values in ``%e`` format.  This string is sent to the
        workers and is also used as the cache key.
    """
    return ",".join("%e" % value for value in values)


class XspecLikelihood:
    r"""Log-likelihood of an XSPEC model, evaluated by a pool of workers.

    The log-likelihood is :math:`\ln L = -S/2`, where :math:`S` is the fit
    statistic of the XCM file (e.g. chi-squared or C-statistic).

    Parameters
    ----------
    xcm : str or path-like
        XCM file containing the data and the model.
    n_procs : int, optional
        Number of worker processes.  Default is 1.
    lmod : dict, optional
        Mapping from local-model package names to their build directories.
    info : XcmInfo, optional
        Result of :func:`read_xcm` for `xcm`.  If not given, the file is read
        again.

    Attributes
    ----------
    labels : list of str
        Labels of the thawed parameters, in the order expected by all
        methods.
    ndim : int
        Number of thawed parameters.
    calls, evaluations : int
        Number of requested points and number of points actually evaluated
        by XSPEC (the difference was answered from the cache).

    Notes
    -----
    Points outside the XSPEC hard limits of any parameter are assigned a
    log-likelihood of ``-inf`` and are not evaluated.

    We recommend using the object as a context manager, so that the workers
    are stopped when it is no longer needed.  Otherwise, call :meth:`close`
    explicitly.

    Examples
    --------
    >>> with XspecLikelihood("fit.xcm", n_procs=8) as like:  # doctest: +SKIP
    ...     lnl = like.log_likelihood(points)
    """

    cache_size = 65536

    def __init__(self, xcm, n_procs=1, lmod=None, info=None):
        if n_procs < 1:
            raise ValueError("n_procs must be positive")
        self.info = info if info is not None else read_xcm(xcm, lmod=lmod)
        self.parameters = list(self.info.parameters)
        self.labels = self.info.labels
        self.lower = np.array([p.hard_min for p in self.parameters])
        self.upper = np.array([p.hard_max for p in self.parameters])
        self.cache = OrderedDict()
        self.calls = 0
        self.evaluations = 0
        self.workers = []
        try:
            for _ in range(n_procs):
                self.workers.append(_Worker(self.info.xcm, self.labels, lmod))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Stop all worker processes."""
        for worker in self.workers:
            worker.close()
        self.workers = []

    @property
    def ndim(self):
        """int: Number of thawed parameters."""
        return len(self.parameters)

    def statistic(self, points):
        """Evaluate the fit statistic at one or more points.

        Parameters
        ----------
        points : array_like, shape (n_points, ndim) or (ndim,)
            Parameter values in physical units, in the order of
            :attr:`labels`.

        Returns
        -------
        numpy.ndarray, shape (n_points,)
            Fit statistic of each point, or NaN for points outside the XSPEC
            hard limits.

        Raises
        ------
        ValueError
            If the number of columns of `points` differs from :attr:`ndim`.
        RuntimeError
            If a worker process exits unexpectedly.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.ndim:
            raise ValueError(f"expected {self.ndim} parameters, got {points.shape[1]}")
        out = np.full(len(points), np.nan)
        inside = np.all((points >= self.lower) & (points <= self.upper), axis=1)
        todo = {}
        for i in np.nonzero(inside)[0]:
            key = encode(points[i])
            if key in self.cache:
                out[i] = self.cache[key]
                self.cache.move_to_end(key)
            else:
                todo.setdefault(key, []).append(i)
        self.calls += len(points)
        self.evaluations += len(todo)
        queue = list(todo)
        by_fd = {w.fileno(): w for w in self.workers}
        free = list(by_fd)
        busy = {}
        while queue or busy:
            while queue and free:
                fd = free.pop()
                key = queue.pop()
                by_fd[fd].send(key)
                busy[fd] = deque([key])
            ready, _, _ = select.select(list(busy), [], [], 0.05)
            for fd in ready:
                for value in by_fd[fd].results():
                    key = busy[fd].popleft()
                    for i in todo[key]:
                        out[i] = value
                    self.cache[key] = value
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                if not busy[fd]:
                    del busy[fd]
                    free.append(fd)
        return out

    def log_likelihood(self, points):
        """Evaluate the log-likelihood at one or more points.

        Parameters
        ----------
        points : array_like, shape (n_points, ndim) or (ndim,)
            Parameter values in physical units.

        Returns
        -------
        numpy.ndarray, shape (n_points,)
            :math:`-S/2` for each point, or ``-inf`` for points outside the
            hard limits.
        """
        stat = self.statistic(points)
        return np.where(np.isfinite(stat), -0.5 * stat, -np.inf)

    def __call__(self, values):
        """Return the log-likelihood of a single parameter vector."""
        return float(self.log_likelihood(values)[0])

    def map(self, _function, points):
        """Evaluate the log-likelihood with the pool interface of pocoMC.

        Parameters
        ----------
        _function : callable
            Ignored.  pocoMC passes its own wrapper of the likelihood.
        points : iterable of array_like
            Parameter vectors.

        Returns
        -------
        numpy.ndarray
            Log-likelihood of each vector.
        """
        return self.log_likelihood(list(points))
