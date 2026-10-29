"""
Removed-reach inflow record for a *manually built* network (points 1 & 4).

The builder accepts generic ``Reach``/``Node`` objects directly, so a network
does NOT have to come from TopAGNPS.  When some reaches are dropped (``ignore``
flag or ``strahler_drop``), the built ``Cche1dNetwork`` carries the same
removed-reach inflow record a TopAGNPS import does, and it can be written to
``*_inflow_reaches.csv``.  These tests exercise that path without a raster and
without the package's heavy dependencies (duck-typed reaches), so they run in a
bare interpreter as well as pixi.

Run:  pixi run -e test test   (or: python3 tests/test_manual_inflow.py)
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = importlib.util.spec_from_file_location(
    "topagnps2cche1d_network", ROOT / "topagnps2cche1d" / "network.py")
net = importlib.util.module_from_spec(SPEC)
sys.modules["topagnps2cche1d_network"] = net
SPEC.loader.exec_module(net)


class _N:
    """Duck-typed generic node: exposes the attributes the builder reads."""

    def __init__(self, id):
        self.id = id
        self.x = None
        self.y = None
        self.usid = None
        self.dsid = None


class _R:
    """Duck-typed generic reach: exposes the attributes the builder reads."""

    def __init__(self, id, receiving_reach_id, pts, ignore=False,
                 strahler=None):
        self.id = id
        self.receiving_reach_id = receiving_reach_id
        self.ignore = ignore
        self.strahler_number = strahler
        nodes = {}
        for k, (x, y) in enumerate(pts):
            n = _N(100 * id + k)
            n.x, n.y = x, y
            if k > 0:
                n.usid = 100 * id + k - 1
            if k + 1 < len(pts):
                n.dsid = 100 * id + k + 1
            nodes[n.id] = n
        ids = list(nodes)
        self.us_nd_id = ids[0]
        self.ds_nd_id = ids[-1]
        self.nodes = nodes


def make_reach(rid, recv, pts, ignore=False, strahler=None):
    return _R(rid, recv, pts, ignore=ignore, strahler=strahler)


def junction_reaches():
    """
    Junction tree, y increases downstream:

        A(1)            C(3)
          \\            /
           B(2) ---- D(4)   (D: outlet)

    A and C drain into B's head; B drains into D's head.
    """
    J = (30.0, 60.0)
    D = (50.0, 100.0)
    return [
        make_reach(1, 2, [(0.0, 0.0), (15.0, 20.0), J], strahler=1),
        make_reach(3, 2, [(30.0, 0.0), J], strahler=1),
        make_reach(2, 4, [J, (40.0, 80.0), D], strahler=2),
        make_reach(4, None, [D, (60.0, 140.0), (70.0, 160.0)], strahler=2),
    ]


def build_and_drop_A():
    reaches = junction_reaches()
    reaches[0].ignore = True          # remove A (one of the two inflows into B)
    net_out = net.build_network_from_reaches(reaches, strahler_drop=0)
    return net_out


def test_manual_build_produces_inflow_reaches_record():
    """
    A hand-built network (no TopAGNPS, no Watershed) that drops reach A must
    still emit the removed-reach inflow record, pointing at B's head node.
    B's head keeps C as a kept inflow, so the confluence is MERGED into one
    continuous channel (a single interior type-6 node at the pixel -- see
    the builder docstring for why a type-3 tail -> type-6 head pair is
    fatal input); the removed reach A is recorded in the inflow record at
    that node (a head only becomes a type-1 inflow BC when ALL its upstream
    reaches are removed -- see the two-removed test below).
    """
    n = build_and_drop_A()
    errors, warnings = net.validate_network(n)
    assert errors == [] and warnings == []

    recs = net.inflow_reaches_records(n)
    # one removed reach (A = 1) feeding kept reach B (2)
    assert [r for r, _ in recs] == [1]

    # the receiving node is B's head (a valid ND_ID)
    head_nd = n.channel_heads[2]
    assert (1, head_nd) in recs
    node = n.node_by_id(head_nd)
    # B still has the kept inflow C, so the confluence is MERGED into one
    # continuous channel: the head is a single interior type-6 node (the
    # C tail and the B head are the same node), and its USID is the
    # predecessor inside the merged channel.
    assert node.nd_type == net.NODE_USERADDED
    assert node.usid != net.NEGATIVE_ONE

    # net.removed_reaches carries the generic-reach form
    assert n.removed_reaches == [(1, 2)]


def test_manual_build_writes_inflow_reaches_csv(tmp_path):
    n = build_and_drop_A()
    out = tmp_path / "manual_inflow_reaches.csv"
    count = net.write_inflow_reaches_csv(n, out)
    assert count == 1
    text = out.read_text().splitlines()
    assert text[0] == "inflow_topagnps_reach_id,receiving_node_id"
    head_nd = n.channel_heads[2]
    assert f"1,{head_nd}" in text


def test_manual_build_no_ignored_has_empty_record():
    n = net.build_network_from_reaches(junction_reaches(), strahler_drop=0)
    assert net.inflow_reaches_records(n) == []
    assert n.removed_reaches == []


def test_two_removed_inflows_record_both():
    reaches = junction_reaches()
    reaches[0].ignore = True          # A
    reaches[1].ignore = True          # C
    n = net.build_network_from_reaches(reaches, strahler_drop=0)
    errors, _ = net.validate_network(n)
    assert errors == []
    recs = net.inflow_reaches_records(n)
    assert sorted(r for r, _ in recs) == [1, 3]
    # B's head now has TWO removed inflows and no kept inflow -> source (0)
    assert n.node_by_id(n.channel_heads[2]).nd_type == net.NODE_SOURCE


def test_removed_reach_inflow_pairs_standalone():
    reaches = junction_reaches()
    reaches[0].ignore = True
    assert net.removed_reach_inflow_pairs(reaches, strahler_drop=0) == [(1, 2)]
    # a removed reach whose receiver is ALSO removed is not listed separately
    reaches2 = junction_reaches()
    reaches2[0].ignore = True
    reaches2[2].ignore = True          # B removed; A's receiver B is removed
    # A feeds B (removed) -> not a direct neighbour of a kept reach
    pairs = net.removed_reach_inflow_pairs(reaches2, strahler_drop=0)
    assert 1 not in [p[0] for p in pairs]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
