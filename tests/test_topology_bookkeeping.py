"""
Fast topology + bookkeeping tests for the CCHE1D build on generic reaches
(no raster, no slow steps -- runs in seconds).

Covers the semantics that the long Ohio/Goodwin e2e runs exercise on real
data:

* head-node classification after coarsening (removed-reach inflow records,
  type 1 vs type 0 heads),
* cross-section bookkeeping through the junction-tail node merge
  (the type-3 tail nodes merged onto a junction head pixel carry the head's
  cross-section profile, and a cell-drainage record maps through the
  bijective origin map to the network node standing at that pixel),
* the single-node (single-pixel) reach expansion,
* 1-point channels and their -1 reach-range link sentinel,
* the tw_cell / inflow_reaches DataFrames.

Run:  pixi run -e test test
"""

import importlib.util
import sys
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
    from topagnps2cche1d.cell import Cell
except ImportError:  # pragma: no cover - bare interpreter
    pytest.skip("topagnps2cche1d package deps not available",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_reach(rid, recv, pts, slope=0.001):
    """Generic reach with a complete US->DS node chain (usid + dsid)."""
    r = Reach(id=rid, receiving_reach_id=recv, slope=slope)
    nodes = []
    for k, (x, y) in enumerate(pts):
        nodes.append(Node(id=100 * rid + k, x=x, y=y))
    for i, n in enumerate(nodes):
        n.dsid = nodes[i + 1].id if i + 1 < len(nodes) else None
        n.usid = nodes[i - 1].id if i > 0 else None
        r.add_node(n)
    r.us_nd_id = nodes[0].id
    r.ds_nd_id = nodes[-1].id
    return r


def small_watershed():
    """
    Junction tree, y increases downstream:

        A(1)            C(1)
          \\            /
           B(2) ---- D(2)   (D: outlet)

    A and C drain into B's head (a 2-way junction); B drains into D's head
    (a trivial 1-way connection).  One cell drains into B's head node.
    """
    J = (30.0, 60.0)
    D = (50.0, 100.0)
    w = Watershed()
    w.add_reach(make_reach(1, 2, [(0.0, 0.0), (15.0, 20.0), J]))
    w.add_reach(make_reach(3, 2, [(30.0, 0.0), J]))
    w.add_reach(make_reach(2, 4, [J, (40.0, 80.0), D]))
    w.add_reach(make_reach(4, None, [D, (60.0, 140.0), (70.0, 160.0)]))
    w.update_graph()
    w.assign_strahler_number_to_reaches()
    head_b = w.reaches[2].nodes[w.reaches[2].us_nd_id]
    w.add_cell(Cell(id=1, type="source", area=123.0,
                    receiving_reach_id=2))
    head_b.inflow_cell_sources.add(1)
    return w


def build(w):
    w.update_watershed()
    w.assign_cross_section_to_all_points()
    return w.build_cche1d_network(strahler_drop=0)


# ---------------------------------------------------------------------------
# full network
# ---------------------------------------------------------------------------

def test_small_network_validates_with_one_junction():
    w = small_watershed()
    net = build(w)
    errors, warnings = netmod.validate_network(net)
    assert errors == [] and warnings == []
    types = [n.nd_type for n in net.nodes]
    assert types.count(2) == 1      # A+C into B: one junction
    assert types.count(9) == 1      # outlet
    junc = [n for n in net.nodes if n.nd_type == 2][0]
    assert len(net.junction_branches(junc.nd_id)) == 2


def test_cell_drainage_at_junction_head_maps_to_junction_node():
    """
    The cell drains into B's head generic node.  In the full network that
    maps onto the type-2 junction head, and the two type-3 tail nodes
    merged onto that same pixel carry the same cross-section profile
    (provenance override to the head's section).
    """
    w = small_watershed()
    net = build(w)
    df_tw = w.create_cche1d_twcells_df()
    assert len(df_tw) == 1
    row = df_tw.iloc[0]
    nd = net.node_by_id(int(row["TW_NDDSID"]))
    assert nd.nd_type == 2
    assert float(row["TW_DRAREA"]) == 123.0
    # the tails merged onto this pixel share the head's cross-section profile
    by_id = {c.cs_id: c for c in net.csecs}
    head_profile = [(p[1], p[2]) for p in by_id[nd.csid].points]
    tails = [t for t in net.nodes if t.nd_type == 3 and t.dsid == nd.nd_id]
    assert len(tails) == 2
    for t in tails:
        t_profile = [(p[1], p[2]) for p in by_id[t.csid].points]
        assert t_profile == head_profile


def test_inflow_reaches_df_empty_when_nothing_removed():
    w = small_watershed()
    build(w)
    assert w.get_list_of_inflow_reaches_df().empty


# ---------------------------------------------------------------------------
# coarsened network (removed-reach inflow record)
# ---------------------------------------------------------------------------

def test_coarsened_junction_head_records_removed_inflow():
    """
    Remove A (one of the two inflows into B's head).  B's head still has C
    as a kept inflow, so it is a merged confluence (type 6) -- never a
    type-1 boundary (a type-1 head of a multi-node link would stop the
    model at import, c1dlnk.f90).  The removed reach A is recorded in the
    inflow_reaches DataFrame at B's head node, and the type-1 inflow-
    boundary label sits on the node just downstream of the head.
    """
    w = small_watershed()
    w.reaches[1].ignore_reach()      # remove A
    w.update_graph()
    net = w.build_cche1d_network(strahler_drop=0)
    errors, warnings = netmod.validate_network(net)
    assert errors == [] and warnings == []
    # the removed reach A is recorded in the inflow_reaches DataFrame
    df_in = w.get_list_of_inflow_reaches_df()
    assert len(df_in) == 1
    row = df_in.iloc[0]
    assert int(row["inflow_topagnps_reach_id"]) == 1
    by_id = {n.nd_id: n for n in net.nodes}
    assert int(row["receiving_node_id"]) in by_id
    # B's head: C is still a kept inflow -> merged confluence (type 6),
    # never a type-1 boundary
    head = by_id[int(row["receiving_node_id"])]
    assert head.nd_type == netmod.NODE_USERADDED
    # the type-1 inflow-boundary label sits on the node just downstream
    # of the head (the removed reach's water enters at the head itself)
    assert by_id[head.dsid].nd_type == netmod.NODE_INFLOW
    # the cell still drains into B's head (unchanged)
    df_tw = w.create_cche1d_twcells_df()
    assert len(df_tw) == 1
    assert int(df_tw.iloc[0]["TW_NDDSID"]) in by_id


def test_two_removed_inflows_make_a_source_head():
    # A and C removed -> B's head has 2 removed upstream reaches ->
    # source (type 0), not an inflow boundary
    w = small_watershed()
    w.reaches[1].ignore_reach()
    w.reaches[3].ignore_reach()
    w.update_graph()
    net = w.build_cche1d_network(strahler_drop=0)
    errors, _ = netmod.validate_network(net)
    assert errors == []
    types = [n.nd_type for n in net.nodes]
    assert 0 in types and 1 not in types
    df_in = w.get_list_of_inflow_reaches_df()
    assert sorted(int(v) for v in df_in["inflow_topagnps_reach_id"]) == [1, 3]


def test_single_node_reach_expansion_after_upstream_removal():
    """
    Linear chain 1 -> 2 -> 3 where reach 2 is a single pixel.  With reach
    1 removed, reach 2's only kept neighbour is 3, so the pair merges into
    one continuous channel.  Reach 2's head (the boundary for the removed
    reach 1) heads a MULTI-NODE link, so it stays a legal source (type 0)
    and the type-1 inflow-boundary label goes on the node just downstream
    of the head (a type-1 multi-node link head would stop the model at
    import, c1dlnk.f90).  The single pixel is NOT expanded here: the
    expansion (Watershed.add_us_and_ds_node_to_single_node_reaches) requires
    both neighbours kept, and reach 2's downstream neighbour is the outlet.
    """
    w = Watershed()
    w.add_reach(make_reach(1, 2, [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]))
    single = Node(id=2001, x=30.0, y=30.0)
    r2 = Reach(id=2, receiving_reach_id=3, slope=0.001)
    r2.add_node(single)
    r2.us_nd_id = r2.ds_nd_id = 2001
    w.add_reach(r2)
    w.add_reach(make_reach(3, None, [(40.0, 40.0), (50.0, 50.0),
                                     (60.0, 60.0)]))
    w.reaches[1].ignore_reach()
    w.update_graph()
    w.assign_strahler_number_to_reaches()
    net = w.build_cche1d_network(strahler_drop=0)
    errors, _ = netmod.validate_network(net)
    assert errors == []
    types = [n.nd_type for n in net.nodes]
    assert 1 in types and 2 not in types
    # the boundary head is a legal source; the type-1 label is one node
    # downstream of it
    by_id = {n.nd_id: n for n in net.nodes}
    head = by_id[1]
    assert head.nd_type == netmod.NODE_SOURCE
    assert by_id[head.dsid].nd_type == netmod.NODE_INFLOW
    df_in = w.get_list_of_inflow_reaches_df()
    assert len(df_in) == 1
    assert int(df_in.iloc[0]["inflow_topagnps_reach_id"]) == 1


def test_single_node_outlet_reach_merges_into_continuous_channel():
    """
    Chain 1 -> 2 where reach 2 is a single pixel AND the outlet reach.
    The outlet channel's head has exactly one kept inflow (channel 1), so
    the builder merges the pair into ONE continuous channel: the single
    pixel is an interior type-6 node and the channel tail (the same node)
    is the watershed outlet (type 9, self-referencing DSID).  There is no
    1-point channel and no -1 reach-range link sentinel as a result.
    """
    w = Watershed()
    w.add_reach(make_reach(1, 2, [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]))
    single = Node(id=2001, x=30.0, y=30.0)
    r2 = Reach(id=2, receiving_reach_id=None, slope=0.001)
    r2.add_node(single)
    r2.us_nd_id = r2.ds_nd_id = 2001
    w.add_reach(r2)
    w.update_graph()
    w.assign_strahler_number_to_reaches()
    net = w.build_cche1d_network(strahler_drop=0)
    errors, _ = netmod.validate_network(net)
    assert errors == []
    assert len(net.channels) == 1
    ch = net.channels[0]
    assert ch.ndusid != ch.nddsid          # merged: one continuous channel
    outlet = net.node_by_id(ch.nddsid)
    assert outlet.nd_type == netmod.NODE_WATSHDOUTL
    assert outlet.dsid == ch.nddsid        # outlet self-reference
    assert all(l.rcusid != netmod.NEGATIVE_ONE for l in net.links)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        netmod.write_cche1d_dat_files(net, d, prefix="onetest")
        back = netmod.read_cche1d_network(d)
        errors2, _ = netmod.validate_network(back)
        assert errors2 == []
        assert all(l.rcusid != netmod.NEGATIVE_ONE for l in back.links)
