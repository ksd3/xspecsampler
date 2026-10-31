"""Searching and downloading observations from the HEASARC archive.

Queries are sent to the Table Access Protocol (TAP) service of HEASARC as ADQL
and are run on the master table of each mission.  The results are parsed with
``astropy.io.votable``, which is installed with the ``archive`` extra.  All
times are in MJD, as given in the master tables.

Examples
--------
::

    from xspecsampler import archive

    obs = archive.query("nustar", ra=299.5903, dec=35.2016, radius=0.25)
    nicer = archive.query("nicer", ra=299.5903, dec=35.2016, radius=0.25)
    pairs = archive.simultaneous(obs, nicer, min_overlap=600)
    url = archive.archive_directory(obs[0])
    archive.download(url, "data/" + obs[0].obsid, include=["*_cl.evt.gz", "*.hk*"])
"""

from __future__ import annotations

import fnmatch
import io
import os
import posixpath
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

TAP_URL = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync"
DATALINK_URL = "https://heasarc.gsfc.nasa.gov/xamin/vo/datalink/{table}?id={row}"
SESAME_URLS = (
    "https://cds.unistra.fr/cgi-bin/nph-sesame/-oI/A?{name}",
    "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oI/A?{name}",
)
USER_AGENT = "xspecsampler-archive"


@dataclass(frozen=True)
class Mission:
    """Master table of a mission and the names of its time columns.

    Attributes
    ----------
    table : str
        Name of the HEASARC master table.
    start : str
        Column with the start time (MJD).
    end : str or None
        Column with the end time (MJD).  If None, the end time is computed
        as ``start + duration / 86400``.
    exposure : str
        Column with the exposure time (s).
    duration : str, optional
        Column with the duration (s), used when `end` is None.
    """
    table: str
    start: str
    end: str | None        # None: end = start + duration_column/86400
    exposure: str
    duration: str | None = None


# Master tables and their time/exposure columns (checked against HEASARC, October 2026).
MISSIONS = {
    "nustar": Mission("numaster", "time", "end_time", "exposure_a"),
    "nicer": Mission("nicermastr", "time", "end_time", "exposure"),
    "swift": Mission("swiftmastr", "start_time", "stop_time", "xrt_exposure"),
    "xmm": Mission("xmmmaster", "time", "end_time", "duration"),
    "chandra": Mission("chanmaster", "time", None, "exposure", duration="exposure"),
    "suzaku": Mission("suzamaster", "time", "stop_time", "exposure"),
    "rxte": Mission("xtemaster", "time", None, "exposure", duration="duration"),
}


@dataclass
class Observation:
    """Row of a mission master table.

    Attributes
    ----------
    mission : str
        Mission name, a key of :data:`MISSIONS`.
    obsid : str
        Observation identifier.
    start, end : float
        Start and end of the observation (MJD).
    exposure : float
        Exposure time in seconds, as given in the master table.
    ra, dec : float
        Pointing direction in degrees.
    row : int
        HEASARC row identifier, used by :func:`data_links`.
    extra : dict
        Additional columns requested with the `columns` argument of
        :func:`query`.
    """
    mission: str
    obsid: str
    start: float          # MJD
    end: float            # MJD
    exposure: float       # s, as in the master table
    ra: float
    dec: float
    row: int              # HEASARC row id, used for data links
    extra: dict = field(default_factory=dict)

    @property
    def duration(self):
        """float: Elapsed time between start and end, in seconds."""
        return (self.end - self.start) * 86400.0


def _get(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def tap_query(adql, timeout=120):
    """Run an ADQL query on the HEASARC TAP service.

    Parameters
    ----------
    adql : str
        ADQL query.
    timeout : float, optional
        Timeout of the request in seconds.  Default is 120.

    Returns
    -------
    astropy.table.Table
        Query result.

    Raises
    ------
    ImportError
        If astropy is not installed.
    RuntimeError
        If the service reports an error.
    """
    try:
        from astropy.io.votable import parse_single_table
    except ImportError as exc:
        raise ImportError("archive queries need astropy: pip install 'xspecsampler[archive]'") from exc
    url = TAP_URL + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": adql})
    data = _get(url, timeout)
    status = re.search(rb'name="QUERY_STATUS" value="(\w+)"[^>]*>([^<]*)', data)
    if status and status.group(1) == b"ERROR":
        raise RuntimeError(f"HEASARC TAP error: {status.group(2).decode(errors='replace').strip()}")
    return parse_single_table(io.BytesIO(data)).to_table(use_names_over_ids=True)


def build_query(mission, ra, dec, radius=0.25, start=None, end=None, where=None, columns=(), limit=None):
    """Build the ADQL query used by :func:`query`.

    The parameters are those of :func:`query`.  This function is public so
    that the query can be inspected or modified before it is run with
    :func:`tap_query`.

    Returns
    -------
    str
        ADQL query.
    """
    m = MISSIONS[mission]
    end_column = m.end or f"({m.start} + {m.duration}/86400.0)"
    cols = ["__row", "obsid", "ra", "dec", f"{m.start} AS t_start", f"{end_column} AS t_end",
            f"{m.exposure} AS t_exposure", *columns]
    conditions = [f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(radius)})) = 1"]
    if start is not None:
        conditions.append(f"{end_column} >= {float(start)}")
    if end is not None:
        conditions.append(f"{m.start} <= {float(end)}")
    if where:
        conditions.append(f"({where})")
    top = f"TOP {int(limit)} " if limit else ""
    return f"SELECT {top}{', '.join(cols)} FROM {m.table} WHERE {' AND '.join(conditions)} ORDER BY {m.start}"


def query(mission, ra, dec, radius=0.25, start=None, end=None, where=None, columns=(), limit=None, timeout=120):
    """Find the observations of a mission pointed near a position.

    Parameters
    ----------
    mission : str
        Mission name, a key of :data:`MISSIONS`.
    ra, dec : float
        Position in degrees (ICRS).
    radius : float, optional
        Search radius in degrees.  Default is 0.25.
    start, end : float, optional
        If given, only observations that overlap the interval from `start`
        to `end` (MJD) are returned.
    where : str, optional
        Additional ADQL condition on the master table, for example
        ``"xrt_expo_wt > 0"`` to select Swift observations with XRT windowed
        timing data, or ``"status = 'archived'"`` for NuSTAR.
    columns : sequence of str, optional
        Additional master-table columns, returned in
        :attr:`Observation.extra`.
    limit : int, optional
        Maximum number of rows.
    timeout : float, optional
        Timeout of the request in seconds.  Default is 120.

    Returns
    -------
    list of Observation
        Matching observations, sorted by start time.

    Raises
    ------
    KeyError
        If the mission is not known.

    Examples
    --------
    >>> obs = query("nustar", 299.5903, 35.2016, where="status = 'archived'")  # doctest: +SKIP
    """
    if mission not in MISSIONS:
        raise KeyError(f"unknown mission {mission!r}; known: {sorted(MISSIONS)}")
    table = tap_query(build_query(mission, ra, dec, radius, start, end, where, columns, limit), timeout=timeout)
    out = []
    for row in table:
        def value(name, cast=float):
            v = row[name]
            try:
                return cast(v)
            except (TypeError, ValueError):
                return float("nan") if cast is float else None
        out.append(Observation(
            mission=mission, obsid=str(row["obsid"]).strip(), start=value("t_start"), end=value("t_end"),
            exposure=value("t_exposure"), ra=value("ra"), dec=value("dec"), row=int(row["__row"]),
            extra={c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in columns},
        ))
    return out


def resolve_name(name, timeout=30):
    """Resolve an object name into coordinates with the CDS Sesame service.

    Sesame queries SIMBAD, NED and VizieR.  A mirror is tried if the main
    service cannot be reached.

    Parameters
    ----------
    name : str
        Object name, e.g. ``"Cyg X-1"``.
    timeout : float, optional
        Timeout of each request in seconds.  Default is 30.

    Returns
    -------
    ra, dec : float
        J2000 coordinates in degrees.

    Raises
    ------
    RuntimeError
        If the name cannot be resolved.  In that case we recommend passing
        the coordinates directly.
    """
    errors = []
    for template in SESAME_URLS:
        try:
            text = _get(template.format(name=urllib.parse.quote(name)), timeout).decode(errors="replace")
        except OSError as exc:
            errors.append(f"{template.split('/')[2]}: {exc}")
            continue
        coordinates = parse_sesame(text)
        if coordinates is not None:
            return coordinates
        errors.append(f"{template.split('/')[2]}: name not found")
    raise RuntimeError(f"could not resolve {name!r} ({'; '.join(errors)}); pass ra/dec instead")


def parse_sesame(text):
    """Extract the coordinates from a Sesame response.

    Parameters
    ----------
    text : str
        Response of Sesame in ``-oI`` format.

    Returns
    -------
    tuple of float or None
        J2000 coordinates in degrees from the first ``%J`` line, or None if
        there is no such line.
    """
    for line in text.splitlines():
        if line.startswith("%J "):
            ra, dec = line.split()[1:3]
            return float(ra), float(dec)
    return None


def search(target, missions=("nustar", "nicer", "swift", "xmm", "chandra"), radius=0.25, **kwargs):
    """Find the observations of several missions pointed near a target.

    Parameters
    ----------
    target : str or tuple of float
        Object name, resolved with :func:`resolve_name`, or ``(ra, dec)`` in
        degrees.
    missions : sequence of str, optional
        Missions to search.  Default is NuSTAR, NICER, Swift, XMM-Newton and
        Chandra.
    radius : float, optional
        Search radius in degrees.  Default is 0.25.
    **kwargs
        Further keyword arguments passed to :func:`query`.

    Returns
    -------
    dict
        Lists of :class:`Observation`, keyed by mission.
    """
    ra, dec = resolve_name(target) if isinstance(target, str) else target
    return {mission: query(mission, ra, dec, radius=radius, **kwargs) for mission in missions}


def simultaneous(primary, others, min_overlap=0.0):
    """Find pairs of observations that overlap in time.

    Parameters
    ----------
    primary, others : sequence of Observation
        Observations to compare.
    min_overlap : float, optional
        Minimum overlap in seconds.  Default is 0.

    Returns
    -------
    list of tuple
        ``(p, o, overlap)`` for each observation ``p`` in `primary` and ``o``
        in `others` that overlap, where ``overlap`` is in seconds.

    Notes
    -----
    The overlap is computed from the start and end times of the observations,
    not from their good time intervals, and is therefore an upper limit on
    the simultaneous exposure.  We recommend using it to select candidate
    observations, and computing the simultaneous exposure from the good time
    intervals of the event files.
    """
    pairs = []
    for p in primary:
        for o in others:
            overlap = (min(p.end, o.end) - max(p.start, o.start)) * 86400.0
            if overlap > 0 and overlap >= min_overlap:
                pairs.append((p, o, overlap))
    return pairs


def data_links(observation, timeout=60):
    """Retrieve the DataLink entries of an observation.

    Parameters
    ----------
    observation : Observation
        Observation returned by :func:`query`.
    timeout : float, optional
        Timeout of the request in seconds.  Default is 60.

    Returns
    -------
    list of dict
        One dictionary per entry, with keys such as ``access_url``,
        ``description`` and ``error_message``.

    Raises
    ------
    ImportError
        If astropy is not installed.
    """
    try:
        from astropy.io.votable import parse_single_table
    except ImportError as exc:
        raise ImportError("archive queries need astropy: pip install 'xspecsampler[archive]'") from exc
    table_name = MISSIONS[observation.mission].table
    data = _get(DATALINK_URL.format(table=table_name, row=observation.row), timeout)
    table = parse_single_table(io.BytesIO(data)).to_table(use_names_over_ids=True)
    return [{c: (str(r[c]) if r[c] is not None else "") for c in table.colnames} for r in table]


def archive_directory(observation, timeout=60):
    """Find the directory of an observation in the HEASARC archive.

    Parameters
    ----------
    observation : Observation
        Observation returned by :func:`query`.
    timeout : float, optional
        Timeout of the request in seconds.  Default is 60.

    Returns
    -------
    str
        HTTPS URL of the directory.

    Raises
    ------
    RuntimeError
        If the DataLink entries contain no archive directory.
    """
    for link in data_links(observation, timeout):
        url = link.get("access_url", "")
        if "/FTP/" in url and not link.get("error_message"):
            return url.replace("//" + observation.obsid, "/" + observation.obsid)
    raise RuntimeError(f"no archive directory found for {observation.mission} {observation.obsid}")


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def list_directory(url, timeout=60):
    """List the contents of an archive directory.

    Parameters
    ----------
    url : str
        URL of the directory.
    timeout : float, optional
        Timeout of the request in seconds.  Default is 60.

    Returns
    -------
    files, directories : list of str
        Absolute URLs of the files and subdirectories.  Sorting links, the
        parent directory and links outside `url` are ignored.
    """
    if not url.endswith("/"):
        url += "/"
    parser = _Links()
    parser.feed(_get(url, timeout).decode(errors="replace"))
    files, directories = [], []
    for href in parser.hrefs:
        if href.startswith("?") or href.startswith("#"):
            continue
        absolute = urllib.parse.urljoin(url, href)
        if not absolute.startswith(url) or absolute == url:
            continue
        (directories if absolute.endswith("/") else files).append(absolute)
    return sorted(set(files)), sorted(set(directories))


def download(url, destination, include=None, exclude=None, timeout=300, dry_run=False):
    """Download an archive directory and its subdirectories.

    Parameters
    ----------
    url : str
        URL of the directory, e.g. from :func:`archive_directory`.
    destination : str or path-like
        Local directory.  It is created if it does not exist.
    include, exclude : sequence of str, optional
        Shell-style patterns matched against the file names, for example
        ``["*_cl.evt.gz", "*.hk.gz"]``.  A file is downloaded if it matches
        a pattern in `include` (or `include` is not given) and no pattern in
        `exclude`.
    timeout : float, optional
        Timeout of each request in seconds.  Default is 300.
    dry_run : bool, optional
        If True, nothing is downloaded.  Default is False.

    Returns
    -------
    list
        Local paths of the files, or, if `dry_run` is True, the URLs that
        would be downloaded.

    Notes
    -----
    Files that already exist with the expected size are not downloaded
    again, so an interrupted download can be resumed by repeating the call.
    We recommend a dry run first, since the full directory of an
    observation can be large.
    """
    destination = Path(destination)
    if not url.endswith("/"):
        url += "/"
    fetched = []
    files, directories = list_directory(url)
    for file_url in files:
        name = posixpath.basename(urllib.parse.urlparse(file_url).path)
        if include and not any(fnmatch.fnmatch(name, p) for p in include):
            continue
        if exclude and any(fnmatch.fnmatch(name, p) for p in exclude):
            continue
        target = destination / name
        if dry_run:
            fetched.append(file_url)
            continue
        request = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
            if target.exists() and size is not None and target.stat().st_size == int(size):
                fetched.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with open(partial, "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(partial, target)
        fetched.append(target)
    for directory_url in directories:
        name = posixpath.basename(urllib.parse.urlparse(directory_url).path.rstrip("/"))
        fetched += download(directory_url, destination / name, include, exclude, timeout, dry_run)
    return fetched
