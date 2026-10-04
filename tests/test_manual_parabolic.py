"""
Manual CCHE1D network test -- no TopAGNPS inputs (item 5).

Builds a small network from a hand-written `Reach`/`Node` definition where every
reach uses a **parabolic cross section** (constant Manning n) and a **constant
slope**, so the cross-section thalweg is adjusted as we go upstream.  Asserts:

* the network builds and validates with no errors and no warnings,
* the parabolic profile (6 points) is present in the built cross sections,
* the thalweg is monotone upstream and the outlet sits at elevation 0,
* a write/read round trip re-validates cleanly.

Fast (no raster).  Run:  pixi run -e test test
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "topagnps2cche1d_network", ROOT / "topagnps2cche1d" / "network.py")
netmod = importlib.util.module_from_spec(SPEC)
sys.modules["topagnps2cche1d_network"] = netmod
SPEC.loader.exec_module(netmod)

try:
    from topagnps2cche1d.watershed import Watershed
    from topagnps2cche1d.node import Node
    from topagnps2cche1d.reach import Reach
except ImportError:  # pragma: no cover - bare interpreter
    pytest.skip("topagnps2cche1d package deps not available",
                allow_module_level=True)

WS = [-14.0, -5.0, -1.0, 1.0, 5.0, 14.0]
ZS = [6.0, 2.0, 0.0, 0.0, 2.0, 6.0]
SLOPES = {1: 0.0008, 3: 0.0006, 2: 0.0005, 4: 0.0010}


def make_reach(rid, recv, pts):
    r = Reach(id=rid, receiving_reach_id=recv, slope=SLOPES[rid])
    nodes = [Node(id=100 * rid + k, x=x, y=y) for k, (x, y) in enumerate(pts)]
    for i, n in enumerate(nodes):
        n.dsid = nodes[i + 1].id if i + 1 < len(nodes) else None
        n.usid = nodes[i - 1].id if i > 0 else None
        r.add_node(n)
    r.us_nd_id, r.ds_nd_id = nodes[0].id, nodes[-1].id
    return r


def build_manual():
    J = (30.0, 60.0)
    D = (50.0, 100.0)
    w = Watershed()
    w.add_reach(make_reach(1, 2, [(0, 0), (15, 20), J]))
    w.add_reach(make_reach(3, 2, [(30, 0), J]))
    w.add_reach(make_reach(2, 4, [J, (40, 80), D]))
    w.add_reach(make_reach(4, None, [D, (60, 140), (70, 160)]))
    w.update_graph()
    w.assign_strahler_number_to_reaches()
    w.assign_cross_section_to_all_points(
        cs_type="user_defined", ws=WS, zs=ZS,
        lfp_idx=1, lbt_idx=2, rbt_idx=5, rfp_idx=6,
        mannings_roughness=0.035)
    w.adjust_cross_sections_elevation_with_slope()
    return w


def test_manual_parabolic_thalweg_monotone_upstream():
    w = build_manual()

    def thalweg(node):
        return w.cross_sections[node.csid].get_thalweg_elevation()

    outlet = w.reaches[4].nodes[w.reaches[4].ds_nd_id]
    assert abs(thalweg(outlet)) < 1e-9
    for rid, r in w.reaches.items():
        chain, cur = [], r.us_nd_id
        while True:
            chain.append(thalweg(r.nodes[cur]))
            if cur == r.ds_nd_id:
                break
            cur = r.nodes[cur].dsid
        assert all(a > b - 1e-9 for a, b in zip(chain, chain[1:])), (rid, chain)


def test_manual_parabolic_builds_validates_and_has_profile():
    w = build_manual()
    net = w.build_cche1d_network(strahler_drop=0)
    errors, warnings = netmod.validate_network(net)
    assert errors == [] and warnings == []
    types = Counter(n.nd_type for n in net.nodes)
    assert types[2] == 1       # the r1/r3 -> r2 junction
    assert types[9] == 1       # the outlet
    # every built cross section carries the 6-point parabolic profile
    assert len(net.csecs) == len(net.nodes)
    for cs in net.csecs:
        assert len(cs.points) == 6
        assert [p[1] for p in cs.points] == [float(v) for v in WS]


def test_manual_parabolic_round_trip(tmp_path):
    w = build_manual()
    net = w.build_cche1d_network(strahler_drop=0)
    w.write_cche1d_dat_files(casename="manual", output_folder=str(tmp_path))
    back = netmod.read_cche1d_network(tmp_path)
    errors, warnings = netmod.validate_network(back)
    assert errors == [] and warnings == []
    assert len(back.nodes) == len(net.nodes)
    # the parabolic profile survives the file round trip
    by_id = {c.cs_id: c for c in back.csecs}
    sample = back.node_by_id(net.nodes[0].nd_id)
    assert [p[1] for p in by_id[sample.csid].points] == [float(v) for v in WS]
