"""Archive search and orbital phases.

The offline tests need nothing; the live test queries HEASARC and is skipped
when astropy is missing or HEASARC cannot be reached.
"""

import math
import urllib.request

import pytest

from xspecsampler import archive, phase
from xspecsampler.archive import Observation

# Brocksopp et al. (1999): P = 5.599829 d, T0 = HJD 2441874.707 (superior conjunction of the black hole)
P, T0 = 5.599829, 41874.207


def obs(obsid, start, end, mission="nustar"):
    return Observation(mission, obsid, start, end, 1000.0, 299.59, 35.20, 1)


def test_orbital_phase_interval():
    # an observation starting at MJD 56231.3535 begins at phase 0.854 of this ephemeris
    p0, p1, mid, span = phase.phase_interval(56231.3535, 56231.7404, P, T0)
    assert abs(p0 - 0.854) < 0.001 and 0 < span < 0.1 and p0 < mid < p1
    assert phase.phase_distance(0.97) == pytest.approx(0.03)
    assert phase.propagated_phase_error(T0 + 100 * P, P, T0, 1e-5, 0.0) == pytest.approx(100 * 1e-5 / P)


def test_simultaneous_overlap_seconds():
    a = [obs("a", 100.0, 100.5)]
    b = [obs("b", 100.4, 101.0, "nicer"), obs("c", 101.0, 102.0, "nicer")]
    pairs = archive.simultaneous(a, b)
    assert [(p.obsid, o.obsid) for p, o, _ in pairs] == [("a", "b")]
    assert math.isclose(pairs[0][2], 0.1 * 86400)
    assert archive.simultaneous(a, b, min_overlap=0.2 * 86400) == []


def test_build_query_uses_mission_columns():
    q = archive.build_query("swift", 1.0, 2.0, radius=0.25, start=50000, end=50010, where="xrt_expo_wt > 0")
    assert "FROM swiftmastr" in q and "stop_time >= 50000.0" in q and "start_time <= 50010.0" in q
    assert "(xrt_expo_wt > 0)" in q
    q = archive.build_query("chandra", 1.0, 2.0)
    assert "(time + exposure/86400.0) AS t_end" in q
    with pytest.raises(KeyError):
        archive.query("einstein", 0, 0)


def test_parse_sesame_and_directory_listing(monkeypatch):
    text = "# Cyg X-1\n#=S=Simbad (via url):    1\n%@ 1\n%J 299.59031591 +35.20160625 = 19:58:21.67 +35:12:05.7\n"
    assert archive.parse_sesame(text) == (299.59031591, 35.20160625)
    assert archive.parse_sesame("# nothing\n") is None
    page = ('<a href="?C=N;O=D">Name</a><a href="/FTP/nustar/">Parent</a>'
            '<a href="event_cl/">event_cl/</a><a href="nu1_cl.evt.gz">x</a><a href="https://elsewhere/">y</a>')
    monkeypatch.setattr(archive, "_get", lambda url, timeout: page.encode())
    files, dirs = archive.list_directory("https://h/FTP/nustar/obs/1")
    assert files == ["https://h/FTP/nustar/obs/1/nu1_cl.evt.gz"]
    assert dirs == ["https://h/FTP/nustar/obs/1/event_cl/"]


def _heasarc_reachable():
    try:
        urllib.request.urlopen("https://heasarc.gsfc.nasa.gov/", timeout=15)
        return True
    except OSError:
        return False


@pytest.mark.network
def test_live_cyg_x1_nustar_nicer_counts():
    pytest.importorskip("astropy")
    if not _heasarc_reachable():
        pytest.skip("HEASARC not reachable")
    ra, dec = 299.5903, 35.2016
    nu = archive.query("nustar", ra, dec, radius=0.25, end=60522.0, columns=("issue_flag",))
    clean = [o for o in nu if str(o.extra["issue_flag"]).strip() in ("0", "0.0")]
    assert len(clean) == 27  # NuSTAR pointings with issue flag 0 up to 2024 July (archive as of October 2026)
    sample = [o for o in clean if o.obsid != "80902318003"]  # drop the 0.2 ks pointing
    ni = archive.query("nicer", ra, dec, radius=0.25, start=min(o.start for o in sample))
    assert len({p.obsid for p, _, _ in archive.simultaneous(sample, ni)}) == 13
    url = archive.archive_directory(next(o for o in sample if o.obsid == "30001011002"))
    assert url.endswith("/nustar/data/obs/00/3/30001011002/")
