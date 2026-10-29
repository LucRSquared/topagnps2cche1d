"""
Manual CCHE1D channel-network builder, writer, validator and reader.

Define a network by listing channels (reaches) as ordered node coordinates
plus the upstream connectivity, then write the six CCHE1D .dat input files
(nodes / reach / channel / link / csec / csprf) that the Fortran reader
(Flow/db_w_ntw_in.f90, Flow/db_w_c1d_in.f90, Flow/db_w_cs_in.f90) accepts.

This module deliberately re-implements the *computing-order* and *junction*
conventions of the existing topagnps2cche1d pipeline
(Watershed.renumber_all_nodes_and_reaches_in_CCHE1D_computational_order and
Watershed._create_junctions_between_reaches) so that networks built by hand
are indistinguishable, from the Fortran's point of view, from networks
imported from TopAGNPS.

Conventions implemented (verified against the Fortran source and the
known-good Goodwin-Creek files in data_inputs/):

* Every file: line 1 = record count, line 2 = header, tab separated,
  1-based integer IDs, ``-1`` means "none".
* ``*_nodes.dat``: 12 values per node
    ND_ID ND_FRMNO ND_TYPE ND_XC ND_YC ND_DSID ND_USID ND_US2ID
    ND_CSID ND_RSID ND_STID ND_SBID
  followed by the junction block
    <number of junction nodes>
    <J> <branch3> <branch4> <branch5>      (branches 1 and 2 are the inline
                                             ND_USID / ND_US2ID)
  ``ND_DSID``/``ND_USID``/``ND_US2ID`` reference **ND_ID**.  ``ND_FRMNO``
  equals ``ND_ID`` and is assigned by the CCHE1D computing-order DFS.
* Computing order: iterative DFS from the outlet channel, visiting each
  channel's inflow channels in descending channel-id order, post-order
  (see tools.custom_dfs_traversal_sorted_predecessors).  Each channel's node
  chain is numbered head -> tail as it is reached; the result is the
  upstream -> downstream "compute" order.  This makes CH_ID == the
  channel's position in that order.
* Junction nodes (ND_TYPE = 2) may have up to **5** upstream branches
  (ND_JUNCT5 in the Fortran).  Branches 3..5 live in the junction block;
  absent branches are written as ``-1`` (the Fortran applies MAX(x,0)).
  Branch order matches the reference
  (``Watershed._create_junctions_between_reaches``) and the known-good files:
  inflow channels sorted by **ascending TopAGNPS-style reach id** fill the
  branches in order -- branch 1 (ND_USID) = tail node of the *smallest-id*
  inflow channel, branch 2 (ND_US2ID) = the next, and so on.
* **Junction branch nodes (ND_TYPE = 3)**: every channel that discharges
  into a junction must end in a type-3 ENDOFLINK node sitting exactly at
  the junction head's pixel, with ``ND_DSID`` pointing at the head; the
  channel's ``CH_NDDSID``/``LK_NDDSID`` is that type-3 node (netchan
  channel 1: 3 -> 4; LoopTest channel 2: 1 -> 8).  When the channel's own
  tail pixel is not the head pixel, the tail keeps its coordinates (user
  geometry is never moved) and a SYNTHETIC type-3 node is inserted at the
  head pixel, with a tail -> branch reach segment between them (netchan
  reach 7: 26 -> 4).  Only when the tail pixel already equals the head
  pixel does the tail itself become the branch node.  The branch node's
  cross section is the junction head's (netchan: nodes 4, 5, 6 share CSID
  4).  The model never uses the coordinates for anything (it rebuilds the
  network from the DSID chains); they only serve the GUI display.
* **Single-inflow confluences merge**: a kept reach whose head has exactly
  one kept inflow is merged with that inflow into ONE continuous channel
  (a split would give a type-3 tail pointing at a type-6 head, which
  c1dlnk.f90 cannot code -- C1DERRF; a 1-branch type-2 junction would make
  cs_adj_jnc.f90 corrupt the thalweg of section 1).  The confluence is one
  shared node when the two pixels coincide, else both nodes are kept at
  their own coordinates and joined by a cross-boundary reach segment.
* **Coarsened (inflow boundary) heads**: a kept head whose upstream reaches
  were all removed is a boundary.  Exactly ONE removed upstream reach ->
  the head is a source (type 0, the only c1dlnk-legal head type for a
  multi-node channel) and the type-1 inflow-boundary label goes on the node
  just DOWNSTREAM of the head (a type-1 head of a multi-node channel stops
  the model at import; the model never reads the label, and the removed-
  reach inflow record points at the head).  A boundary head whose link is a
  single node (and has no stronger functional type 3/9) gets the type-1
  head itself -- legal for a 1-node link (c1dlnk.f90:157-158).  >= 2 removed
  upstream reaches -> plain source (type 0).  This is the original
  reference pipeline's placement
  (``Watershed._find_removed_reaches_inflow_nodes``).
* ``*_reach.dat``: one row per segment between two consecutive nodes;
  ``RC_NDUSID``/``RC_NDDSID`` carry the real connectivity, ``RC_ID`` is an
  identifier.
* ``*_channel.dat``: one row per whole channel (its first and last node).
* ``*_link.dat``: ``LK_ID`` = CH_ID, ``LK_CMPSEQ`` = the channel's position in
  computing order, ``LK_NDUSID``/``LK_NDDSID`` = channel head/tail nodes,
  ``LK_RCUSID``/``LK_RCDSID`` = first/last reach of the channel.
* ``*_csec.dat`` rows follow the header column order
    CS_ID CS_NPTS CS_LOB CS_LBT CS_ROB CS_RBT [CS_SVTYPE CS_TYPE ...]
  (the Fortran reads the six leading values positionally; its internal
  CS_ROB/CS_RBT variables are swapped, but identically for GUI- and
  topagnps2cche1d-generated files, so matching the header order is correct).
* ``*_csprf.dat``: one row per cross-section point
    CP_ID CP_CSID CP_POSIDX CP_W CP_Z CP_RGH CP_BLOCK,
  points ordered by increasing CP_W (station).

Only the standard library is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Node types (cche1d_gui-vs2019/MyHeader.h, NDTYPE_*)
NODE_SOURCE = 0        # head of a source channel
NODE_INFLOW = 1        # inflow boundary (upstream reaches removed)
NODE_JUNCTION = 2      # confluence of >= 2 inflow branches (max 5)
NODE_END_LINK = 3      # end of a link joining a junction
NODE_STRTUS = 4        # start of US (auto)
NODE_AUTOADDED = 5     # auto added
NODE_USERADDED = 6     # user added (interior / trivial-connection heads)
NODE_WATSHDOUTL = 9    # watershed outlet
NODE_STRTDS = 10       # start of DS
NODE_BRIDGE = 11
NODE_CULVERT = 12
NODE_DROP = 13
NODE_MEASFLUME = 14
NODE_DAM = 15
NODE_JUNCTSOURCE = 20  # junction source (loop support)
NODE_JUNCTEND = 21     # junction end (loop support)

VALID_NODE_TYPES = {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21}
MAX_JUNCTION_BRANCHES = 5
NEGATIVE_ONE = -1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NodeSpec:
    """A single CCHE1D node (all 12 CCHE1D columns)."""

    nd_id: int
    frmno: int
    nd_type: int
    xc: float
    yc: float
    dsid: int
    usid: int
    us2id: int
    csid: int = NEGATIVE_ONE
    rsid: int = NEGATIVE_ONE
    stid: int = 1
    sbid: int = 1


@dataclass
class ReachSpec:
    """A single CCHE1D reach: one segment between two nodes."""

    rc_id: int
    ndusid: int
    nddsid: int
    order: int
    length: float


@dataclass
class ChannelSpec:
    """A CCHE1D channel: the whole chain between its first and last node."""

    ch_id: int
    ndusid: int
    nddsid: int


@dataclass
class LinkSpec:
    lk_id: int
    cmpseq: int
    ndusid: int
    nddsid: int
    rcusid: int
    rcdsid: int
    lk_type: int = 1


@dataclass
class CrossSectionSpec:
    """A cross section with its profile points (posidx, w, z, rgh, block)."""

    cs_id: int
    lob: int
    lbt: int
    rob: int
    rbt: int
    points: list = field(default_factory=list)


@dataclass
class JunctionExtra:
    """3rd..5th inflow branches of a junction node (beyond USID/US2ID)."""

    node_id: int
    branches345: tuple  # (branch3, branch4, branch5), -1 = absent


@dataclass
class Cche1dNetwork:
    """A complete, CCHE1D-ready channel network."""

    nodes: list
    reaches: list
    channels: list
    links: list
    csecs: list
    junctions: list = field(default_factory=list)
    name: str = "network"
    # metadata only (never written): nd_id -> generic node id, so a caller
    # can map a network node back to the generic reach node it came from
    # (used to assign per-node cross sections).  A junction tail node
    # (ND_TYPE 3) merged onto a receiving head points at the receiving
    # head's generic node (the cross section standing at that pixel).
    node_provenance: dict = field(default_factory=dict)
    # Bijective variant: nd_id -> the generic node at this channel's own
    # point (no tail override).  Every kept generic node maps to exactly one
    # network node, so callers that attribute *bookkeeping* (cell drainage,
    # removed-reach records) to a node use this one.  EXCEPTION: a merged
    # confluence node carries a TWO-element list [inflow-tail generic id,
    # receiver-head generic id] (both generic nodes are the same pixel), so
    # its value is iterable -- flatten before inverting.
    node_origin: dict = field(default_factory=dict)
    # generic reach id -> head ND_ID, for every channel in the built network.
    # Lets a caller map the channel a removed reach feeds into onto the CCHE1D
    # node (its head) where that water enters the kept network.
    channel_heads: dict = field(default_factory=dict)
    # [(removed_reach_id, receiving_reach_id)] for reaches dropped by the
    # ``ignore`` flag / Strahler threshold.  Populated by
    # ``build_network_from_reaches``; empty for a pure manual build with no
    # ignored reaches.  See ``inflow_reaches_records`` for the node-id form.
    removed_reaches: list = field(default_factory=list)

    def node_by_id(self, nd_id: int) -> NodeSpec:
        return next(n for n in self.nodes if n.nd_id == nd_id)

    def junction_branches(self, nd_id: int) -> list:
        """All inflow branch node ids of a node, in branch order 1..5."""
        n = self.node_by_id(nd_id)
        branches = [n.usid, n.us2id if n.us2id != NEGATIVE_ONE else NEGATIVE_ONE]
        for extra in self.junctions:
            if extra.node_id == nd_id:
                branches.extend(extra.branches345)
                break
        return [b for b in branches if b != NEGATIVE_ONE]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _interpolate(points, step):
    """Resample a polyline at constant spacing, keeping endpoints."""
    if len(points) < 2 or step is None or step <= 0:
        return list(points)
    seg = [_dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(seg)
    if total <= 0:
        return list(points)
    n = max(2, int(math.ceil(total / step)) + 1)
    cum = [0.0]
    for L in seg:
        cum.append(cum[-1] + L)
    out = []
    for i in range(n):
        s = total * i / (n - 1)
        j = 1
        while j < len(seg) and cum[j] < s:
            j += 1
        t = (s - cum[j - 1]) / (seg[j - 1] or 1.0)
        x = points[j - 1][0] + t * (points[j][0] - points[j - 1][0])
        y = points[j - 1][1] + t * (points[j][1] - points[j - 1][1])
        out.append((x, y))
    return out


def cche1d_compute_order(receiving, outlet):
    """CCHE1D computing order of channels.

    ``receiving`` maps channel_id -> receiving channel_id (or None for the
    outlet).  Returns channel ids upstream -> downstream, exactly as
    ``tools.custom_dfs_traversal_sorted_predecessors`` does for the
    TopAGNPS graph (iterative DFS from the outlet, inflow channels in
    descending id order, post-order).
    """
    preds = {}
    for r, recv in receiving.items():
        if recv is not None and recv != r:
            preds.setdefault(recv, []).append(r)

    visited, result = set(), []
    stack = [outlet]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            pre = sorted(preds.get(node, []), reverse=True)
            stack.extend(p for p in pre if p not in visited)
            result.append(node)
    return result[::-1]


# ---------------------------------------------------------------------------
# Manual network definition (the "final product" API)
# ---------------------------------------------------------------------------

def cross_section_to_spec(cs, cs_id=None):
    """
    Convert a topagnps2cche1d ``CrossSection`` (or any object exposing the
    same duck-typed attributes) into a ``CrossSectionSpec`` for this module.

    The generic cross section stores ``ws``/``zs`` left-to-right and four
    1-based feature indices (lfp, lbt, rbt, rfp).  This maps them onto the
    CCHE1D csec columns exactly as
    ``Watershed.create_cche1d_csec_csprf_df`` does:

        CS_LOB = lfp_idx    CS_LBT = lbt_idx
        CS_ROB = rfp_idx    CS_RBT = rbt_idx

    ``mannings_roughness`` may be a scalar (broadcast to every point) or a
    per-point sequence.  Points are emitted in ``ws`` order (already left to
    right) with ``CP_BLOCK`` = 1, matching the reference pipeline.  No numpy
    is required: the arrays are consumed by iteration / ``float()``.
    """
    ws = [float(w) for w in cs.ws]
    zs = [float(z) for z in cs.zs]
    n = len(ws)
    if n != len(zs):
        raise ValueError(
            f"cross section: ws ({n}) and zs ({len(zs)}) length mismatch")
    if n == 0:
        raise ValueError("cross section has no points")
    n_rgh = cs.n_rgh
    if isinstance(n_rgh, (int, float)):
        n_rgh_list = [float(n_rgh)] * n
    else:
        n_rgh_list = [float(v) for v in n_rgh]
    if len(n_rgh_list) != n:
        raise ValueError(
            f"cross section: roughness length {len(n_rgh_list)} != {n} points")
    if cs_id is None:
        cs_id = getattr(cs, "id", NEGATIVE_ONE)
    points = [(i + 1, ws[i], zs[i], n_rgh_list[i], 1) for i in range(n)]
    return CrossSectionSpec(
        cs_id=cs_id,
        lob=int(cs.lfp_idx), lbt=int(cs.lbt_idx),
        rob=int(cs.rfp_idx), rbt=int(cs.rbt_idx),
        points=points)


def _reaches_to_defs(reaches):
    """
    Convert an iterable of generic ``Reach`` objects into
    ``(defs, node_ids)`` where:

    * ``defs``     maps ``reach_id -> (receiving_id, [(x, y), ...])`` (points
      in US -> DS order), and
    * ``node_ids`` maps ``reach_id -> [generic_node_id, ...]`` in the same
      US -> DS order.

    ``node_ids`` lets a caller map a renumbered network node back to the
    generic reach node it was created from (see ``Cche1dNetwork
    .node_provenance``).  The outlet normalisation (a dangling
    ``receiving_reach_id`` maps to ``None``) is applied here.
    """
    reaches = list(reaches)
    ids = {reach.id for reach in reaches}
    defs = {}
    node_ids = {}
    for reach in reaches:
        nodes = reach.nodes
        order = []
        nid_list = []
        cur = nodes[reach.us_nd_id]
        while cur is not None:
            order.append((cur.x, cur.y))
            gid = getattr(cur, "id", None)
            if gid is not None:
                nid_list.append(gid)
            if cur.dsid is None or cur.dsid not in nodes:
                break
            cur = nodes[cur.dsid]
        if len(order) < 1:
            raise ValueError(
                f"reach {reach.id}: no nodes in US->DS order "
                f"(check us_nd_id / dsid chain)")
        recv = reach.receiving_reach_id
        if recv is not None and recv not in ids:
            recv = None
        defs[reach.id] = (recv, order)
        node_ids[reach.id] = nid_list
    return defs, node_ids


def channel_defs_from_reaches(reaches):
    """
    Build the ``{id: (receiving_id, [(x, y), ...])}`` mapping that
    ChannelNetworkBuilder accepts, from the generic ``Reach``/``Node`` objects
    of topagnps2cche1d (or anything duck-typed like them).

    Each reach must expose ``.id``, ``.receiving_reach_id``, ``.us_nd_id`` and
    ``.nodes`` (dict id -> node); each node must expose ``.x``, ``.y``, ``.id``
    and ``.dsid``.  Nodes are read in US -> DS order by walking the ``dsid``
    chain — the same convention as ``Reach.get_nodes_us_ds_order``.

    Outlet normalisation: a reach whose ``receiving_reach_id`` is ``None`` **or
    not present among the provided reaches** (TopAGNPS marks the outlet with a
    dangling id) is treated as the outlet channel and mapped to ``None``.
    """
    defs, _node_ids = _reaches_to_defs(reaches)
    return defs


def _kept_ids(reaches, strahler_drop):
    """
    Return the set of reach ids that survive coarsening: a reach is kept when
    it is not flagged ``ignore`` and its Strahler order is above
    ``strahler_drop``.  A reach with no Strahler order (a hand-built manual
    network) is kept unless ignored.  This is the single keep rule shared by
    ``build_network_from_reaches`` and ``removed_reach_inflow_pairs`` so the
    kept sub-network and the removed-reach inflow record never disagree.
    """
    reaches = (list(reaches.values()) if isinstance(reaches, dict)
               else list(reaches))
    kept = set()
    for r in reaches:
        if getattr(r, "ignore", False):
            continue
        s = getattr(r, "strahler_number", None)
        if s is None or s > strahler_drop:
            kept.add(r.id)
    return kept


def removed_reach_inflow_pairs(reaches, strahler_drop=0):
    """
    Return ``[(removed_reach_id, receiving_reach_id)]`` for every reach that
    is dropped by the ``ignore`` flag or the Strahler threshold but still
    feeds the kept sub-network (its receiver is kept).  This is the generic,
    source-agnostic counterpart of the head-node classification in
    ``build_network_from_reaches``: it needs only the reach objects and the
    shared keep rule, so a hand-built (manual) network gets the same
    removed-reach inflow record as one imported from TopAGNPS.  Only the
    removed reaches that are *direct* upstream neighbours of a kept reach are
    listed (a removed reach whose receiver is itself removed is absorbed into
    that receiver's record) — the same convention as
    ``Watershed._find_removed_reaches_inflow_nodes``.
    """
    reaches = (list(reaches.values()) if isinstance(reaches, dict)
               else list(reaches))
    kept = _kept_ids(reaches, strahler_drop)
    pairs = [(r.id, r.receiving_reach_id) for r in reaches
             if r.id not in kept and r.receiving_reach_id in kept]
    return sorted(pairs, key=lambda p: p[0])


def inflow_reaches_records(net):
    """
    Return ``[(removed_reach_id, receiving_node_id)]`` for the removed reaches
    of a built ``Cche1dNetwork``, where ``receiving_node_id`` is the CCHE1D
    head ``ND_ID`` of the kept channel each removed reach feeds into.

    Works for any network produced by ``build_network_from_reaches`` (which
    stores ``net.removed_reaches`` and ``net.channel_heads``).  A pure manual
    build with no ignored reaches returns ``[]``.  This is the manual-path
    counterpart of ``Watershed.get_list_of_inflow_reaches_df``; both describe
    the same coarsened inflow boundary conditions (the set of AnnAGNPS reach
    outputs that must be generated and wired in as CCHE1D inflows).
    """
    pairs = list(net.removed_reaches) if net.removed_reaches else []
    if not pairs:
        return []
    heads = net.channel_heads
    out = []
    for removed_id, receiving_id in pairs:
        node_id = heads.get(receiving_id)
        if node_id is None:
            raise ValueError(
                f"removed reach {removed_id} feeds kept reach {receiving_id}, "
                "which has no head node in the built network")
        out.append((removed_id, node_id))
    return sorted(out, key=lambda p: p[0])


def build_network_from_reaches(reaches, strahler_drop=0, resample_step=None,
                               cs_template=None, cs_by_generic_id=None):
    """
    Build a CCHE1D network from generic ``Reach``/``Node`` objects, with
    optional Strahler coarsening.  This is the single high-level entry point
    that unifies the old ``Watershed`` CCHE1D build on ``network.py``.

    Parameters
    ----------
    reaches : iterable of Reach (or dict id->Reach)
        Generic reach objects (``.id``, ``.receiving_reach_id``,
        ``.strahler_number``, ``.us_nd_id``, ``.nodes``).  Every reach must
        already have **at least two** nodes in a complete US->DS ``dsid``
        chain -- run ``Watershed.add_us_and_ds_node_to_single_node_reaches``
        first if the source may contain single-pixel reaches.
    strahler_drop : int, default 0
        Drop every reach whose Strahler order is ``<= strahler_drop``.  0
        keeps the whole network.  Because Strahler order is monotone
        downstream (a reach is never higher order than its receivers), a kept
        reach's receiver is either kept or the outlet -- the kept set is
        always a valid connected sub-network with exactly one outlet.
    resample_step : float or None, passed to the builder.
    cs_template : CrossSection, 4-tuple, or None, passed to the builder.
    cs_by_generic_id : dict {generic_node_id: CrossSection}, optional
        Per-node cross sections.  Each node in the built network is mapped
        back to the generic node it was created from (see
        ``Cche1dNetwork.node_provenance``) and given the section registered
        for that node.  Nodes with no registered section fall back to
        ``cs_template``.  When both are ``None`` the network has no cross
        sections (the caller can add them afterwards).

    Head-node classification of the kept sub-network (the old code's
    ``_set_inflow_nodes_type`` / ``_find_removed_reaches_inflow_nodes``,
    re-derived without networkx and without the fragile ``dsid`` walk that
    crashed on aggressive coarsening):

    * ``>= 2`` kept upstream reaches  -> junction (type 2, wired branches,
      one type-3 branch node per inflow at the head pixel)
    * exactly 1 kept upstream reach   -> MERGED with that reach into one
      continuous channel (the confluence is a single interior type-6 node
      when the pixels coincide, else two interior nodes joined by a reach
      segment; see ``ChannelNetworkBuilder.build`` for why)
    * exactly 1 removed upstream reach (no kept inflows) -> source head
      (type 0) with the type-1 inflow-boundary label on the node just
      downstream of the head (the model never reads the label; the
      removed-reach inflow record points at the head).  A type-1 head of a
      multi-node channel would stop the model at import (c1dlnk.f90); for a
      1-node link the head itself carries the type-1 label.
    * ``>= 2`` removed upstream reaches -> source (type 0)
    * no upstream reaches at all   -> source (type 0)
    * exactly 1 removed upstream reach AT a merged/junction head -> head
      type unchanged, type-1 label on the node just downstream of the head

    Returns a ``Cche1dNetwork`` (see ``write_cche1d_dat_files``).
    """
    if isinstance(reaches, dict):
        reaches = list(reaches.values())
    else:
        reaches = list(reaches)
    by_id = {r.id: r for r in reaches}
    # A reach with no Strahler order (a hand-built manual network that never
    # ran assign_strahler_number_to_reaches) is treated as order 1 for the
    # RC_ORDER column, matching the builder's default.
    strahler = {r.id: (getattr(r, "strahler_number", None) or 1)
                for r in reaches}
    # A reach is built when it is not flagged ignore AND its Strahler order is
    # above the drop threshold.  The two coarsening mechanisms compose: the
    # generic Watershed coarsens via the ``ignore`` flag (see
    # Watershed.ignore_reaches_with_strahler_leq), while a standalone caller can
    # coarsen purely by ``strahler_drop``.  Removed reaches are still passed in
    # so a kept reach's head can be classified (one removed inflow -> inflow
    # boundary, several -> source).  The keep rule is shared with
    # removed_reach_inflow_pairs via _kept_ids.
    kept = _kept_ids(reaches, strahler_drop)
    if len(kept) < 2:
        raise ValueError(
            f"strahler_drop={strahler_drop} leaves {len(kept)} reaches; "
            "a CCHE1D network needs at least two")

    # head classification (see docstring): NODE_INFLOW marks a head with
    # EXACTLY ONE removed upstream reach -- the builder places the type-1
    # inflow-boundary label on the node just downstream of the head (a
    # type-1 head of a multi-node channel would stop the model at import);
    # NODE_SOURCE marks a boundary head with >= 2 removed upstream reaches.
    head_types = {}
    for rid in kept:
        inflows = {s for s in by_id
                   if by_id[s].receiving_reach_id == rid}
        kept_in = inflows & kept
        rem_in = inflows - kept
        if len(rem_in) == 1:
            head_types[rid] = NODE_INFLOW
        elif len(rem_in) >= 2 and not kept_in:
            head_types[rid] = NODE_SOURCE

    builder = ChannelNetworkBuilder(
        [by_id[r] for r in kept], strahler=strahler,
        resample_step=resample_step, cs_template=cs_template,
        head_types=head_types, cs_by_generic_id=cs_by_generic_id)
    net = builder.build()
    # Record the removed reaches that feed the kept network, so the caller can
    # emit the removed-reach inflow record (inflow_reaches_records /
    # write_inflow_reaches_csv) without re-deriving it from the generic state.
    # This is what lets a hand-built (manual) network keep its inflow bookkeeping.
    net.removed_reaches = removed_reach_inflow_pairs(reaches, strahler_drop)
    return net


class ChannelNetworkBuilder:
    """
    Build a CCHE1D network from a manual topology definition.

    Parameters
    ----------
    channels : dict {reach_id: (receiving_reach_id_or_None, [(x, y), ...])}
        OR an iterable of generic ``Reach`` objects from topagnps2cche1d
        (see ``channel_defs_from_reaches``).  Each entry defines one channel
        by the id of the reach it discharges into (``None`` for the outlet
        channel) and an ordered list of (x, y) points from its upstream
        (head) end to its downstream (tail) end.  The reach ids should be
        the *TopAGNPS-style* ids (ascending from upstream to downstream), as
        in the original import; the builder re-derives the CCHE1D
        channel/compute order from them.
    strahler : dict {reach_id: strahler_number}, optional (default 1).
    resample_step : float or None
        If given, interior nodes are sampled along each channel geometry at
        this spacing (in coordinate units).
    cs_template : tuple (lob, lbt, rob, rbt), a full ``CrossSection``, or None
        If given, every non-junction, non-outlet node gets a cross section.
        A 4-tuple produces a single bed point (as before).  A full
        ``CrossSection`` object (see ``cross_section_to_spec``) is applied
        identically to every node -- use ``build()`` then assign per-node
        sections to vary them.
    head_type : int, default NODE_SOURCE
        ND_TYPE for the head node of a channel with no upstream inflows.
    head_types : dict {reach_id: nd_type}, optional
        Per-channel marker for coarsened heads (upstream reaches removed).
        ``build_network_from_reaches`` passes NODE_INFLOW for a head with
        exactly ONE removed upstream reach and NODE_SOURCE for a boundary
        head with several.  The builder translates NODE_INFLOW into a
        c1dlnk-legal pattern (module docstring, "Coarsened heads"): a
        boundary head becomes type 0 (type 1 for a 1-node channel) and the
        type-1 inflow-boundary label goes on the node just downstream of
        the head; at a merged/junction head only the label is placed.
        NODE_SOURCE sets a boundary head to type 0.  Channels not listed
        here use the normal head/junction classification.
    """

    def __init__(self, channels, strahler=None, resample_step=None,
                 cs_template=None, head_type=NODE_SOURCE, head_types=None,
                 cs_by_generic_id=None):
        if hasattr(channels, "__iter__") and not isinstance(channels, dict):
            # an iterable of generic Reach objects (see docstring)
            self.channel_defs, self.node_ids = _reaches_to_defs(channels)
        else:
            self.channel_defs = channels
            self.node_ids = {}
        if not self.channel_defs:
            raise ValueError("channels dict is empty")
        self.strahler = strahler or {}
        self.resample_step = resample_step
        self.cs_template = cs_template
        self.cs_by_generic_id = dict(cs_by_generic_id) if cs_by_generic_id else {}
        self.head_type = head_type
        self.head_types = dict(head_types) if head_types else {}

    # -- internals ----------------------------------------------------------
    def _geom(self, rid):
        return _interpolate(self.channel_defs[rid][1], self.resample_step)

    def build(self) -> Cche1dNetwork:
        defs = self.channel_defs
        raw_ids = list(defs)
        # the outlet channel is the one with receiving == None
        outlets = [r for r in raw_ids if defs[r][0] is None]
        if len(outlets) != 1:
            raise ValueError(
                f"expected exactly one outlet channel (receiving=None), "
                f"got {outlets}")
        outlet = outlets[0]
        for r in raw_ids:
            recv = defs[r][0]
            if recv is not None and recv not in defs:
                raise ValueError(f"channel {r} discharges into unknown {recv}")

        # CCHE1D channel ids = positions in the computing-order DFS.
        # cche1d_compute_order expects {id: receiving_id} (recv, not the
        # (recv, points) tuple stored in self.channel_defs).
        receiving = {r: defs[r][0] for r in raw_ids}
        compute_order = cche1d_compute_order(receiving, outlet)
        ch_id = {r: i for i, r in enumerate(compute_order, start=1)}
        geom = {r: self._geom(r) for r in raw_ids}
        npts = {r: max(len(geom[r]), 1) for r in raw_ids}
        # inflows of each channel, in branch order: ASCENDING raw reach id
        # (== TopAGNPS numbering) -- smallest id first, matching the
        # reference code and the known-good files (netchan junction 6:
        # USID=5 / US2ID=4, i.e. the smaller-id tributary first).
        inflows_of = {r: sorted(c for c in raw_ids if defs[c][0] == r)
                      for r in raw_ids}

        # ---- junctions: type-3 branch nodes at the junction head pixel ----
        # Ground truth (the netchan / LoopTest reference files and
        # c1dlnk.f90): every channel c that discharges into a proper
        # junction (a head with >= 2 inflows) must END in a type-3 ENDOFLINK
        # node sitting exactly at the junction head's pixel, with ND_DSID
        # pointing at the head; the junction's branches are those type-3
        # nodes, and the channel's CH_NDDSID/LK_NDDSID is the type-3 node
        # (netchan channel 1: 3 -> 4; LoopTest channel 2: 1 -> 8).  When c's
        # own tail pixel is NOT the head pixel, the tail node keeps its
        # coordinates (user-defined geometry is never moved) and a
        # SYNTHETIC type-3 node is inserted at the head pixel between the
        # tail and the head, with the tail -> branch reach segment between
        # them (netchan reaches 26->...->26->4, LoopTest 2->8).  Only when
        # the tail pixel already equals the head pixel does the tail itself
        # become the branch node.
        insert_tail = {}
        shared_tail = {}
        for c in raw_ids:
            recv = defs[c][0]
            if recv is None or len(inflows_of[recv]) < 2:
                continue
            if geom[c][-1] == geom[recv][0]:
                shared_tail[c] = recv
            else:
                insert_tail[c] = recv

        # ---- single-inflow confluences: MERGE, never split ---------------
        # A kept reach whose head has exactly one kept inflow cannot be
        # represented as two channels: the tributary tail (type 3) would
        # point at a type-6 head, and c1dlnk.f90 has no link code for a
        # link ending downstream of a type-6 node (C1DERRF stops the model
        # at import).  A 1-branch type-2 junction imports but makes
        # cs_adj_jnc.f90 adjust the thalweg of network section 1 (IU(1) =
        # FRMND(-1) -> ABS -> 1), silently corrupting an unrelated section.
        # So the two reaches are merged into ONE continuous channel and the
        # model flows water through it (it rebuilds the network from the
        # node DSID chains; channel grouping is bookkeeping only).  The
        # confluence is a single shared node when the two pixels coincide
        # (a TopAGNPS raster confluence); otherwise BOTH nodes are kept at
        # their own coordinates and joined by a cross-boundary reach
        # segment -- again, user geometry is never moved.  merge_into maps
        # receiver -> its single kept inflow.
        merge_into = {r: inflows_of[r][0] for r in raw_ids
                      if len(inflows_of[r]) == 1}
        shared_head = {r for r in merge_into
                       if geom[merge_into[r]][-1] == geom[r][0]}
        shared_inflow = {merge_into[r] for r in shared_head}

        # ---- number nodes head->tail in computing order; ND_ID == FRMNO ----
        # A shared confluence reuses the inflow reach's tail node number
        # (computed earlier -- DFS is upstream first).  An inserted branch
        # node gets the next number right after its channel's chain, so the
        # channel's FRMNO span stays contiguous (c1dlnk.f90:113 computes
        # ILNCHR(L,1) = LNKEND - LNKSTT + 1) and the outlet channel's tail
        # keeps the last number (NumNodes = LNKEND(LK_COUNT), c1dlnk.f90:296).
        frmno = {}
        inserted = {}
        counter = 0
        for r in compute_order:
            for k in range(npts[r]):
                if k == 0 and r in shared_head:
                    # shared confluence node: reuse the inflow reach's tail
                    frmno[(r, 0)] = frmno[(merge_into[r],
                                           npts[merge_into[r]] - 1)]
                    continue
                counter += 1
                frmno[(r, k)] = counter
            if r in insert_tail:
                counter += 1
                inserted[r] = counter
        if (counter != sum(npts.values()) - len(shared_head)
                + len(insert_tail)):
            raise RuntimeError("node count mismatch (unreachable channel?)")

        # provenance: each (reach, k) maps to the generic node it was created
        # from, so a caller can relate a renumbered network node back to the
        # original reach node (used to carry over per-node cross sections).
        node_origin = {}
        node_provenance = {}
        for (r, k), nd_id in frmno.items():
            nids = self.node_ids.get(r)
            if nids is not None and k < len(nids):
                node_origin[nd_id] = nids[k]
                node_provenance[nd_id] = nids[k]
        # A branch node sits at the junction head's pixel, so its cross
        # section is the head's (netchan: the head and all its type-3
        # branches share one cross-section profile -- CSID 4 for nodes 4,5,6).
        # Inserted nodes map to the head's generic node; a tail that is
        # itself the branch (shared pixel) overrides its own origin.
        # node_origin (the bijective bookkeeping map) never lists inserted
        # nodes, so its bijection with the kept generic nodes is preserved.
        for c in insert_tail:
            recv_nids = self.node_ids.get(insert_tail[c])
            if recv_nids:
                node_provenance[inserted[c]] = recv_nids[0]
        for c in shared_tail:
            recv_nids = self.node_ids.get(shared_tail[c])
            if recv_nids:
                node_provenance[frmno[(c, npts[c] - 1)]] = recv_nids[0]
        # a SHARED merged confluence node carries BOTH generic nodes (the
        # inflow reach's tail and the receiver's head are the same pixel),
        # so cell drainage bookkeeping finds it from either reach.
        for r in shared_head:
            nd_id = frmno[(r, 0)]
            src_nids = self.node_ids.get(merge_into[r])
            head_nids = self.node_ids.get(r)
            if src_nids and head_nids:
                node_origin[nd_id] = [src_nids[-1], head_nids[0]]

        nodes = {}
        for (r, k), nd_id in frmno.items():
            if nd_id in nodes:
                # shared confluence node (created at the inflow reach's
                # tail, identical pixel) -- created once, coordinates kept
                continue
            x, y = geom[r][k]
            nodes[nd_id] = NodeSpec(
                nd_id=nd_id, frmno=nd_id, nd_type=NODE_USERADDED,
                xc=x, yc=y, dsid=NEGATIVE_ONE, usid=NEGATIVE_ONE,
                us2id=NEGATIVE_ONE)
        for c, e in inserted.items():
            # synthetic branch node: at the receiving junction head's pixel
            hx, hy = geom[insert_tail[c]][0]
            nodes[e] = NodeSpec(
                nd_id=e, frmno=e, nd_type=NODE_USERADDED,
                xc=hx, yc=hy, dsid=NEGATIVE_ONE, usid=NEGATIVE_ONE,
                us2id=NEGATIVE_ONE)

        # ---- intra-channel links + tail nodes ----------------------------
        heads = {r: frmno[(r, 0)] for r in raw_ids}
        for r in raw_ids:
            n = npts[r]
            tail = frmno[(r, n - 1)]
            for k in range(n - 1):
                a, b = frmno[(r, k)], frmno[(r, k + 1)]
                nodes[a].dsid = b
                nodes[b].usid = a
            recv = defs[r][0]
            if recv is None:
                # outlet channel: tail is the watershed outlet (terminal node)
                nodes[tail].nd_type = NODE_WATSHDOUTL
                nodes[tail].dsid = tail
            elif merge_into.get(recv) == r:
                # single-inflow confluence: merged into one channel, no
                # branch node.  Shared pixel: the one confluence node is
                # wired by the two chains above.  Distinct pixels: both
                # nodes keep their own coordinates; the cross-boundary
                # wiring (and reach segment, added in the reach loop)
                # joins tail -> head.
                if r not in shared_inflow:
                    h = frmno[(recv, 0)]
                    nodes[tail].dsid = h
                    nodes[h].usid = tail
            elif r in shared_tail:
                # tail pixel == junction head pixel: the tail node IS the
                # junction's branch node (type 3)
                nodes[tail].nd_type = NODE_END_LINK
                nodes[tail].dsid = heads[recv]
            else:
                # junction, distinct tail pixel: the synthetic type-3 branch
                # node stands at the head pixel; the tail keeps its
                # coordinates and its interior (type-6) node type
                e = inserted[r]
                nodes[e].nd_type = NODE_END_LINK
                nodes[e].usid = tail
                nodes[e].dsid = heads[recv]
                nodes[tail].dsid = e

        # head node type + junction wiring
        junctions = []
        for r in raw_ids:
            inflows = inflows_of[r]
            head = frmno[(r, 0)]
            if not inflows:
                # boundary head: no upstream inflow in the (kept) set.  A
                # coarsened network marks these via the head_types override
                # (NODE_INFLOW for one removed upstream reach, NODE_SOURCE
                # for several); otherwise fall back to the default head_type.
                nt = self.head_types.get(r, self.head_type)
                if nt == NODE_INFLOW:
                    # exactly one removed upstream reach.  A type-1 head is
                    # only legal for a 1-NODE link (c1dlnk.f90:157-158); a
                    # type-1 head of a multi-node link stops the model at
                    # import.  The link this head starts is r's own channel
                    # extended through any downstream merge chain (and the
                    # synthetic branch node, when present):
                    g = r
                    while (defs[g][0] is not None and defs[g][0] in merge_into
                           and merge_into[defs[g][0]] == g):
                        g = defs[g][0]
                    link_tail = inserted[g] if g in inserted \
                        else frmno[(g, npts[g] - 1)]
                    if link_tail == head:
                        # 1-node link: type 1 is legal; a functional type
                        # already set by the tail wiring (3 branch / 9
                        # outlet) wins over the label
                        if nodes[head].nd_type == NODE_USERADDED:
                            nodes[head].nd_type = NODE_INFLOW
                    else:
                        # multi-node link: legal source head; the type-1
                        # inflow-boundary label goes on the node just
                        # downstream of the head -- the original reference
                        # pipeline's placement (the model never reads the
                        # label; the removed-reach inflow record points at
                        # the head).  A functional type at that node wins.
                        nodes[head].nd_type = NODE_SOURCE
                        nxt = nodes[head].dsid
                        if (nxt not in (NEGATIVE_ONE, head)
                                and nodes[nxt].nd_type == NODE_USERADDED):
                            nodes[nxt].nd_type = NODE_INFLOW
                else:
                    nodes[head].nd_type = nt
                nodes[head].usid = NEGATIVE_ONE
                continue
            # branch order: inflows by ASCENDING raw reach id (== TopAGNPS
            # numbering), smallest id first -> branch 1 (USID), branch 2
            # (US2ID), then 3..5.  A branch is the channel's synthetic type-3
            # node when one was inserted, else its own tail node.
            ends = [inserted[c] if c in inserted
                    else frmno[(c, npts[c] - 1)] for c in inflows]
            if len(ends) > MAX_JUNCTION_BRANCHES:
                raise ValueError(
                    f"junction node {head} has {len(ends)} inflows; "
                    f"CCHE1D supports at most {MAX_JUNCTION_BRANCHES}")
            if len(ends) == 1:
                # single kept inflow: MERGED with this channel (merge_into).
                # The head is the interior confluence node (shared node or
                # the receiver's own head point); the chain wiring already
                # set its USID and DSID.  No ENDOFLINK tail, no branch, no
                # junction-block row.
                if self.head_types.get(r) == NODE_INFLOW:
                    # one removed upstream reach: the type-1 label goes on
                    # the node just downstream of the head (a functional
                    # type there wins)
                    nxt = nodes[head].dsid
                    if (nxt not in (NEGATIVE_ONE, head)
                            and nodes[nxt].nd_type == NODE_USERADDED):
                        nodes[nxt].nd_type = NODE_INFLOW
                continue
            for e in ends:
                nodes[e].dsid = head
            # proper junction
            nodes[head].nd_type = NODE_JUNCTION
            nodes[head].usid = ends[0]
            nodes[head].us2id = ends[1]
            if self.head_types.get(r) == NODE_INFLOW:
                # one removed upstream reach at a junction head: the type-1
                # label goes on the node just downstream of the head (a
                # functional type there wins)
                nxt = nodes[head].dsid
                if (nxt not in (NEGATIVE_ONE, head)
                        and nodes[nxt].nd_type == NODE_USERADDED):
                    nodes[nxt].nd_type = NODE_INFLOW
            # every type-2 node gets a junction-block row; branches 3..5
            # beyond the inflows are padded with -1 (Fortran MAX(x,0))
            junctions.append(JunctionExtra(
                head, tuple((ends[2:] + [-1, -1, -1])[:3])))

        # ---- channels, reaches, links (in computing order) ----------------
        # A merged pair (inflow T -> receiver M) is ONE continuous channel
        # from T's head to M's tail: splitting it would create a link ending
        # at the shared type-6 confluence node, which c1dlnk.f90 cannot
        # assign a link code for (C1DERRF).  Channel grouping is bookkeeping
        # only -- the model rebuilds the network from the node DSID chains.
        #
        # A channel that ends at a junction with a distinct tail pixel ends
        # at its synthetic type-3 branch node (netchan: CH_NDDSID is the
        # type-3 node, e.g. channel 1: 3 -> 4), and its last reach segment
        # is the tail -> branch segment.  A merged confluence with distinct
        # pixels contributes one cross-boundary segment (tail -> head); its
        # RC_ORDER is the downstream member's (the model does not read
        # RC_ORDER; it is bookkeeping).
        group_of = {}
        for r in raw_ids:
            g = r
            while (defs[g][0] is not None and defs[g][0] in merge_into
                   and merge_into[defs[g][0]] == g):
                g = defs[g][0]
            group_of[r] = g

        channels, reaches, links = [], [], []
        rc_id = 0
        ch_seq = 0
        for r in compute_order:
            if group_of[r] != r:
                continue          # merged into r's group's downstream member
            # group members, upstream -> downstream
            members = [r]
            h = r
            while h in merge_into:
                s = merge_into[h]
                if defs[s][0] != h:
                    break
                members.insert(0, s)
                h = s
            ch_seq += 1
            head = frmno[(members[0], 0)]
            last = members[-1]
            # channel tail: the synthetic branch node when the channel ends
            # at a junction with a distinct tail pixel, else the last node
            tail = inserted[last] if last in inserted \
                else frmno[(last, npts[last] - 1)]
            channels.append(ChannelSpec(ch_seq, head, tail))
            first_rc = rc_id + 1
            for i, m in enumerate(members):
                for k in range(npts[m] - 1):
                    rc_id += 1
                    a, b = frmno[(m, k)], frmno[(m, k + 1)]
                    length = math.hypot(
                        nodes[b].yc - nodes[a].yc, nodes[b].xc - nodes[a].xc)
                    reaches.append(ReachSpec(
                        rc_id, a, b, self.strahler.get(m, 1), length))
                # cross-boundary segment of a distinct-pixel merged
                # confluence (between member i's tail and member i+1's head)
                if i + 1 < len(members):
                    nxt = members[i + 1]
                    a = frmno[(m, npts[m] - 1)]
                    b = frmno[(nxt, 0)]
                    if a != b:
                        rc_id += 1
                        length = math.hypot(
                            nodes[b].yc - nodes[a].yc,
                            nodes[b].xc - nodes[a].xc)
                        reaches.append(ReachSpec(
                            rc_id, a, b, self.strahler.get(nxt, 1), length))
            # final segment: channel tail -> synthetic branch node
            if last in inserted:
                rc_id += 1
                a = frmno[(last, npts[last] - 1)]
                length = math.hypot(
                    nodes[tail].yc - nodes[a].yc,
                    nodes[tail].xc - nodes[a].xc)
                reaches.append(ReachSpec(
                    rc_id, a, tail, self.strahler.get(last, 1), length))
            if head == tail:
                # 1-point channel (unexpanded boundary single-node reach):
                # zero reach segments.  -1 is the CCHE1D absent sentinel; the
                # model ignores LK_RCUSID/LK_RCDSID entirely (it reconstructs
                # the network from the node references).
                links.append(LinkSpec(ch_seq, ch_seq, head, tail,
                                      NEGATIVE_ONE, NEGATIVE_ONE, 1))
            else:
                links.append(LinkSpec(ch_seq, ch_seq, head, tail,
                                      first_rc, rc_id, 1))

        # ---- cross sections ------------------------------------------------
        # Every node carries a cross section, matching the reference pipeline
        # (junction heads and the outlet node included).  Resolution order for
        # each node: a per-node CrossSection looked up by its provenance (the
        # generic node id), else the shared template.  A full CrossSection
        # (ws/zs/n_rgh + 4 indices) is converted once per source and reused;
        # a 4-tuple template keeps the single-bed-point behaviour.
        csecs = []
        have_cs = bool(self.cs_by_generic_id) or self.cs_template is not None
        if have_cs:
            tpl = self.cs_template
            if hasattr(tpl, "ws") and hasattr(tpl, "lfp_idx"):
                template_spec = cross_section_to_spec(tpl, cs_id=NEGATIVE_ONE)
            elif tpl is not None:
                lob, lbt, rob, rbt = tpl
                template_spec = CrossSectionSpec(
                    cs_id=NEGATIVE_ONE, lob=lob, lbt=lbt, rob=rob, rbt=rbt,
                    points=[(1, 0.0, 0.0, 0.03, 1)])
            else:
                template_spec = None
            cache = {}
            for nd in sorted(nodes.values(), key=lambda n: n.nd_id):
                gid = node_provenance.get(nd.nd_id)
                src = self.cs_by_generic_id.get(gid) if gid is not None else None
                if src is None:
                    spec = template_spec
                else:
                    if id(src) not in cache:
                        cache[id(src)] = cross_section_to_spec(src,
                                                               cs_id=NEGATIVE_ONE)
                    spec = cache[id(src)]
                if spec is None:
                    continue
                nd.csid = nd.nd_id
                csecs.append(CrossSectionSpec(
                    cs_id=nd.nd_id,
                    lob=spec.lob, lbt=spec.lbt, rob=spec.rob, rbt=spec.rbt,
                    points=[tuple(p) for p in spec.points]))

        return Cche1dNetwork(
            nodes=sorted(nodes.values(), key=lambda n: n.frmno),
            reaches=reaches, channels=channels, links=links,
            csecs=csecs, junctions=junctions,
            node_provenance=node_provenance, node_origin=node_origin,
            channel_heads=dict(heads),
            name=f"manual_{len(raw_ids)}ch")


# ---------------------------------------------------------------------------
# Writers (exact Fortran formats)
# ---------------------------------------------------------------------------

NODE_HEADER = ("ND_ID", "ND_FRMNO", "ND_TYPE", "ND_XC", "ND_YC", "ND_DSID",
               "ND_USID", "ND_US2ID", "ND_CSID", "ND_RSID", "ND_STID", "ND_SBID")
REACH_HEADER = ("RC_ID", "RC_NDUSID", "RC_NDDSID", "RC_ORDER", "RC_LENGTH")
CHANNEL_HEADER = ("CH_ID", "CH_NDUSID", "CH_NDDSID")
LINK_HEADER = ("LK_ID", "LK_CMPSEQ", "LK_NDUSID", "LK_NDDSID",
               "LK_RCUSID", "LK_RCDSID", "LK_TYPE")
CSEC_HEADER = ("CS_ID", "CS_NPTS", "CS_LOB", "CS_LBT", "CS_ROB", "CS_RBT",
               "CS_SVTYPE", "CS_TYPE", "CS_ORIGIN", "CS_STATION")
CSPRF_HEADER = ("CP_ID", "CP_CSID", "CP_POSIDX", "CP_W", "CP_Z",
                "CP_RGH", "CP_BLOCK")


def _fmt_float(v):
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10f}".rstrip("0").rstrip(".")


def _write_table(path, header, rows):
    lines = [str(len(rows)), "\t".join(header)]
    lines += ["\t".join(str(c) for c in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cche1d_dat_files(net: Cche1dNetwork, out_dir, prefix=None):
    """Write the six CCHE1D input files. Returns {role: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or net.name

    # nodes: 12 values per row, then the junction block (one row per type-2
    # node; branches 3..5 with -1 for absent).  Rows are in ND_ID order: the
    # Fortran reader fills its arrays positionally and uses every node
    # reference as an array index, so the file invariant is ND_ID(I) == I.
    rows = []
    for n in sorted(net.nodes, key=lambda n: n.nd_id):
        rows.append([n.nd_id, n.frmno, n.nd_type, _fmt_float(n.xc),
                     _fmt_float(n.yc), n.dsid, n.usid, n.us2id,
                     n.csid, n.rsid, n.stid, n.sbid])
    # one junction-block row per type-2 node (branch order 3,4,5; -1 =
    # absent), in ND_ID order: the Fortran reader stores branch 3..5 under
    # the row's ND_ID and reads them back by node-file row index (db_w_ntw_
    # in.f90:287-293), so the row's J must equal the row number.  FRMNO
    # order would corrupt the block whenever ND_ID != FRMNO.
    junc_nodes = sorted(net.junctions, key=lambda j: j.node_id)
    junc_lines = [str(len(junc_nodes))]
    for j in junc_nodes:
        b = tuple((j.branches345 + (-1, -1, -1))[:3])
        junc_lines.append(f"{j.node_id}\t{b[0]}\t{b[1]}\t{b[2]}")
    nodes_lines = [str(len(rows)), "\t".join(NODE_HEADER)]
    nodes_lines += ["\t".join(str(c) for c in r) for r in rows]
    nodes_lines += junc_lines
    Path(out_dir / f"{prefix}_nodes.dat").write_text(
        "\n".join(nodes_lines) + "\n", encoding="utf-8")

    reach_rows = [(r.rc_id, r.ndusid, r.nddsid, r.order, _fmt_float(r.length))
                  for r in sorted(net.reaches, key=lambda r: r.rc_id)]
    _write_table(out_dir / f"{prefix}_reach.dat", REACH_HEADER, reach_rows)

    ch_rows = [(c.ch_id, c.ndusid, c.nddsid)
               for c in sorted(net.channels, key=lambda c: c.ch_id)]
    _write_table(out_dir / f"{prefix}_channel.dat", CHANNEL_HEADER, ch_rows)

    lk_rows = [(l.lk_id, l.cmpseq, l.ndusid, l.nddsid, l.rcusid, l.rcdsid,
                l.lk_type) for l in sorted(net.links, key=lambda l: l.lk_id)]
    _write_table(out_dir / f"{prefix}_link.dat", LINK_HEADER, lk_rows)

    # csec columns follow the header order (CS_LOB CS_LBT CS_ROB CS_RBT).
    cs_rows = [(cs.cs_id, len(cs.points), cs.lob, cs.lbt, cs.rob, cs.rbt)
               for cs in sorted(net.csecs, key=lambda c: c.cs_id)]
    cp_rows = []
    cp_id = 0
    for cs in sorted(net.csecs, key=lambda c: c.cs_id):
        for (posidx, w, z, rgh, block) in cs.points:
            cp_id += 1
            cp_rows.append((cp_id, cs.cs_id, posidx, _fmt_float(w),
                            _fmt_float(z), _fmt_float(rgh), block))
    _write_table(out_dir / f"{prefix}_csec.dat", CSEC_HEADER, cs_rows)
    _write_table(out_dir / f"{prefix}_csprf.dat", CSPRF_HEADER, cp_rows)

    return {
        "nodes": out_dir / f"{prefix}_nodes.dat",
        "reach": out_dir / f"{prefix}_reach.dat",
        "channel": out_dir / f"{prefix}_channel.dat",
        "link": out_dir / f"{prefix}_link.dat",
        "csec": out_dir / f"{prefix}_csec.dat",
        "csprf": out_dir / f"{prefix}_csprf.dat",
    }


def write_inflow_reaches_csv(net: Cche1dNetwork, path):
    """
    Write the removed-reach inflow record for a built network to ``path`` as a
    two-column CSV.  The columns are ``inflow_topagnps_reach_id`` (the id of the
    removed reach — the AnnAGNPS/TopAGNPS reach id in the imported case, or the
    reach id you used in a manual build) and ``receiving_node_id`` (the CCHE1D
    head ``ND_ID`` of the kept channel each removed reach feeds).  This is the
    same schema ``Watershed.get_list_of_inflow_reaches_df`` writes, so the
    manual and TopAGNPS paths are byte-consistent.  The rows are whatever
    ``inflow_reaches_records`` returns (empty for a pure manual build with no
    ignored reaches); the file always carries the header line.  Returns the
    number of removed-reach rows written.
    """
    rows = inflow_reaches_records(net)
    Path(path).write_text(
        "inflow_topagnps_reach_id,receiving_node_id\n" +
        "".join(f"{a},{b}\n" for a, b in rows),
        encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------------
# File reader (mirrors the Fortran list-directed READ statements)
# ---------------------------------------------------------------------------

def _read_block(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    count = int(lines[0].strip())
    header = lines[1].split("\t")
    rows = [ln.split("\t") for ln in lines[2:2 + count]]
    return count, header, rows


def read_cche1d_nodes_dat(path):
    """Read a *_nodes.dat file like db_w_ntw_in.f90.
    Returns (nodes, junctions)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    count = int(lines[0].strip())
    nodes = []
    for ln in lines[2:2 + count]:
        t = ln.split("\t")
        nodes.append(NodeSpec(
            nd_id=int(t[0]), frmno=int(t[1]), nd_type=int(t[2]),
            xc=float(t[3]), yc=float(t[4]), dsid=int(t[5]), usid=int(t[6]),
            us2id=int(t[7]), csid=int(t[8]), rsid=int(t[9]),
            stid=int(t[10]), sbid=int(t[11])))
    idx = 2 + count
    n_junc = int(lines[idx].strip())
    junctions = []
    for k in range(n_junc):
        t = lines[idx + 1 + k].split("\t")
        junctions.append(JunctionExtra(
            node_id=int(t[0]),
            branches345=(int(t[1]), int(t[2]), int(t[3]))))
    return nodes, junctions


def read_cche1d_network(prefix_dir):
    """Rebuild a Cche1dNetwork from the six .dat files in ``prefix_dir``.
    The prefix is inferred from the ``*_nodes.dat`` filename.  csec columns
    are stored under their header names (CS_ROB = file col 5, CS_RBT = file
    col 6); note the Fortran swaps these two internally, but identically for
    every file it reads, so this keeps a write->read round-trip faithful."""
    prefix_dir = Path(prefix_dir)
    nodes_file = next(prefix_dir.glob("*_nodes.dat"))
    prefix = nodes_file.name[: -len("_nodes.dat")]

    nodes, junctions = read_cche1d_nodes_dat(prefix_dir / f"{prefix}_nodes.dat")
    _, _, reach_rows = _read_block(prefix_dir / f"{prefix}_reach.dat")
    reaches = [ReachSpec(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                         float(r[4])) for r in reach_rows]
    _, _, ch_rows = _read_block(prefix_dir / f"{prefix}_channel.dat")
    channels = [ChannelSpec(int(r[0]), int(r[1]), int(r[2])) for r in ch_rows]
    _, _, lk_rows = _read_block(prefix_dir / f"{prefix}_link.dat")
    links = [LinkSpec(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                      int(r[4]), int(r[5]), int(r[6])) for r in lk_rows]

    csecs = []
    cs_path = prefix_dir / f"{prefix}_csec.dat"
    if cs_path.exists():
        _, _, cs_rows = _read_block(cs_path)
        csec_map = {}
        for r in cs_rows:
            cs = CrossSectionSpec(cs_id=int(r[0]), lob=int(r[2]), lbt=int(r[3]),
                                  rob=int(r[4]), rbt=int(r[5]), points=[])
            csec_map[cs.cs_id] = cs
            csecs.append(cs)
        cp_path = prefix_dir / f"{prefix}_csprf.dat"
        if cp_path.exists():
            _, _, cp_rows = _read_block(cp_path)
            for r in cp_rows:
                csec_map[int(r[1])].points.append(
                    (int(r[2]), float(r[3]), float(r[4]), float(r[5]),
                     int(r[6])))
        for cs in csecs:
            cs.points.sort(key=lambda p: p[1])

    return Cche1dNetwork(nodes=nodes, reaches=reaches, channels=channels,
                         links=links, csecs=csecs, junctions=junctions,
                         name=prefix)


# ---------------------------------------------------------------------------
# Validation (reverse-engineered from the Fortran import checks)
# ---------------------------------------------------------------------------

class NetworkValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("Network validation failed:\n" +
                         "\n".join(f"  - {e}" for e in errors))


def validate_network(net: Cche1dNetwork):
    """
    Check a Cche1dNetwork against what the CCHE1D Fortran reader expects
    (db_w_ntw_in.f90 / db_w_c1d_in.f90 / db_w_cs_in.f90).

    Returns ``(errors, warnings)``.  ``errors`` are fatal: the Fortran import
    would stop or produce a broken model, so this function raises
    ``NetworkValidationError`` whenever the list is non-empty.  ``warnings``
    mirror things the Fortran import *tolerates* -- most notably a node whose
    ND_CSID has no matching row in the csec file (``db_w_c1d_in.f90:713``
    runs ``DO CS = 1, CS_COUNT / IF(CS_ID(CS).EQ.NDCSID)`` with no else
    branch and silently continues) and cross-section geometry problems
    (``CP_W`` ordering, missing profile rows).  A warnings-only result means
    "the model will import this; the geometry may be incomplete".
    """
    errors, warnings = [], []
    nodes = {n.nd_id: n for n in net.nodes}
    reaches = {r.rc_id: r for r in net.reaches}
    channels = {c.ch_id: c for c in net.channels}
    csecs = {c.cs_id: c for c in net.csecs}

    if not nodes:
        errors.append("network has no nodes")
    if len(nodes) != len(net.nodes):
        errors.append("duplicate ND_IDs")
    if len(channels) != len(net.channels):
        errors.append("duplicate CH_IDs")

    node_ids = set(nodes)
    for n in net.nodes:
        if n.nd_type not in VALID_NODE_TYPES:
            errors.append(f"node {n.nd_id}: invalid ND_TYPE {n.nd_type}")
        for attr in ("dsid", "usid", "us2id"):
            v = getattr(n, attr)
            if v not in (NEGATIVE_ONE, n.nd_id) and v not in node_ids:
                errors.append(f"node {n.nd_id}: {attr}={v} not a known node")
        if n.csid != NEGATIVE_ONE and n.csid not in csecs:
            # db_w_c1d_in.f90:713 tolerates an unmatched ND_CSID (no else
            # branch) -- warning only.
            warnings.append(f"node {n.nd_id}: ND_CSID={n.csid} has no "
                            "matching row in the csec file (tolerated by "
                            "the model; cross-section geometry will be "
                            "missing at this node)")

    # computing order: ND_FRMNO must be a permutation of 1..N and every
    # inflow branch must be computed strictly earlier than its junction node.
    frmno = {n.nd_id: n.frmno for n in net.nodes}
    if set(frmno.values()) != set(range(1, len(frmno) + 1)):
        errors.append("ND_FRMNO is not a permutation of 1..N")
    for n in net.nodes:
        for b in (n.usid, n.us2id):
            if b == NEGATIVE_ONE:
                continue
            if b in frmno and frmno[b] >= n.frmno:
                errors.append(
                    f"node {n.nd_id} (FRMNO {n.frmno}) receives inflow from "
                    f"node {b} (FRMNO {frmno[b]}) not computed earlier")

    # junctions
    for j in net.junctions:
        if j.node_id not in nodes:
            errors.append(f"junction references unknown node {j.node_id}")
            continue
        n = nodes[j.node_id]
        if n.nd_type != NODE_JUNCTION:
            errors.append(f"junction node {j.node_id} has ND_TYPE {n.nd_type}, "
                          f"expected {NODE_JUNCTION}")
        allb = [b for b in (n.usid, n.us2id) + tuple(j.branches345)
                if b != NEGATIVE_ONE]
        if n.usid == NEGATIVE_ONE:
            errors.append(f"junction node {j.node_id}: ND_USID (branch 1) is -1")
        if len(allb) > MAX_JUNCTION_BRANCHES:
            errors.append(f"junction node {j.node_id} has {len(allb)} inflows "
                          f"(max {MAX_JUNCTION_BRANCHES})")
        if len(set(allb)) != len(allb):
            errors.append(f"junction node {j.node_id}: duplicate inflow {allb}")
        for b in allb:
            if b in nodes and nodes[b].dsid != j.node_id:
                errors.append(f"junction node {j.node_id}: inflow branch "
                              f"{b} has DSID {nodes[b].dsid}, expected "
                              f"{j.node_id}")

    # ---- c1dlnk.f90 link-code rules (fatal in the model) ------------------
    # For every link the model reads the type of the node downstream of the
    # link tail (NDCODE(NDDSND(LNKEND))) and maps it to a link code:
    #   type 2 -> 3 (channel junction), 9 -> 2 (watershed outlet),
    #   11..15 -> structures, 3 -> 2 only for the last link in CMPSEQ,
    #   10 -> re-read the tail's own type (structure),
    # anything else -> C1DERRF (the model stops at import).

    # Loop-channel end (documents/LoopTest, the model's own ground truth):
    # a link that starts at a type-2 junction and ends at a type-3 node with
    # ND_DSID = -1.  FRMND is allocated from -1 so FRMND(-1) = -1 is safe;
    # c1dlnk.f90 then evaluates NDCODE(-1) -- strictly out of bounds, but the
    # pattern is what the model authors ship and run for loop testing.  The
    # builder never emits it; here it is a warning (tolerated ground truth),
    # anywhere else it is an error.
    loop_tails = set()
    for l in net.links:
        head = nodes.get(l.ndusid)
        tail = nodes.get(l.nddsid)
        if (head is not None and tail is not None
                and head.nd_type == NODE_JUNCTION
                and tail.nd_type == NODE_END_LINK
                and tail.dsid == NEGATIVE_ONE):
            loop_tails.add(l.nddsid)
    for n in net.nodes:
        if n.dsid == NEGATIVE_ONE:
            if n.nd_id in loop_tails:
                warnings.append(
                    f"node {n.nd_id}: ND_DSID=-1 (loop-channel end, the "
                    "documents/LoopTest pattern); c1dlnk.f90 evaluates "
                    "NDCODE(-1) out of bounds for it -- the model tolerates "
                    "this ground-truth pattern, the builder never emits it")
            else:
                errors.append(
                    f"node {n.nd_id}: ND_DSID=-1 makes the model evaluate "
                    "NDCODE(-1) out of bounds in c1dlnk.f90; the outlet uses "
                    "the self-reference (ND_DSID = own ND_ID)")

    # Looped-channel nodes (20/21) are valid CCHE1D input -- the GUI produces
    # them and db_w_ntw_in.f90:296-305 / c1dlnk.f90:145-185 wire them -- but
    # this builder can never emit them, so one in a built network is suspect.
    for n in net.nodes:
        if n.nd_type in (NODE_JUNCTSOURCE, NODE_JUNCTEND):
            warnings.append(
                f"node {n.nd_id}: ND_TYPE {n.nd_type} (looped channel, a "
                "GUI-produced pattern); the builder does not emit these")

    # A type-2 junction with exactly one branch imports (link code 3) but
    # cs_adj_jnc.f90 computes IU(1) = FRMND(US2ID) = FRMND(-1) = -1,
    # ABS(-1) = 1, and adjusts the thalweg of network section 1 -- a silent
    # geometry corruption unless the junction sits on channel 1.
    for n in net.nodes:
        if n.nd_type != NODE_JUNCTION:
            continue
        branches = net.junction_branches(n.nd_id)
        if len(branches) == 1:
            errors.append(
                f"junction node {n.nd_id} has exactly one inflow branch "
                f"({branches[0]}); the model imports it but cs_adj_jnc.f90 "
                "would corrupt the thalweg of network section 1")

    # link-tail termination rule (c1dlnk.f90:202-278)
    last_cmpseq = max((l.cmpseq for l in net.links), default=0)
    for l in net.links:
        tail = nodes.get(l.nddsid)
        if tail is None or tail.dsid == NEGATIVE_ONE:
            continue          # covered above / by the reference checks
        target = nodes.get(tail.dsid)
        if target is None:
            continue          # dangling DSID already reported
        ttype = target.nd_type
        if ttype == NODE_STRTDS:
            # ICOD == 10: c1dlnk.f90 re-reads the tail's own type
            ttype = tail.nd_type
        is_last = l.cmpseq == last_cmpseq
        ok = (ttype in (NODE_JUNCTION, NODE_WATSHDOUTL, NODE_BRIDGE,
                        NODE_CULVERT, NODE_DROP, NODE_MEASFLUME, NODE_DAM)
              or (ttype == NODE_END_LINK and is_last))
        if not ok:
            errors.append(
                f"link {l.lk_id}: downstream of tail node {l.nddsid} is "
                f"node {tail.dsid} with ND_TYPE {ttype}; c1dlnk.f90 has no "
                "link code for it and the model stops at import (C1DERRF)")

    # c1dlnk.f90:125-163 head-side rule: ILNCHR(L,3) (number of upstream
    # nodes) is assigned from the type of the link's FIRST node --
    #   type 0 source -> 0, type 2 junction -> 2 (+ block branches),
    #   type 20 (loop) -> left at default, type 10 -> 1,
    #   1-NODE link -> 1 (any type is legal there),
    #   anything else on a multi-node link -> C1DERRF (model stops).
    # In particular a type-1 (inflow boundary) node is only legal as the
    # head of a 1-node link -- the builder therefore places the type-1
    # label one node downstream of a coarsened head.
    for l in net.links:
        head = nodes.get(l.ndusid)
        if head is None or l.ndusid == l.nddsid:
            continue          # dangling head / 1-node link: covered/any type
        if head.nd_type not in (NODE_SOURCE, NODE_JUNCTION, NODE_STRTDS,
                                NODE_JUNCTSOURCE):
            errors.append(
                f"link {l.lk_id}: channel head node {l.ndusid} has ND_TYPE "
                f"{head.nd_type}; c1dlnk.f90 can only start a multi-node "
                "channel at a source (0), junction (2), STRTDS (10) or "
                "JUNCTSOURCE (20) node and the model stops at import "
                "(C1DERRF)")

    # c1dlnk.f90:296: handleNTW%NumNodes = LNKEND(LK_COUNT) -- the model
    # sizes its node array from the LAST link's tail FRMNO, so that tail
    # must carry FRMNO N (the node count).  The builder keeps the outlet
    # channel (last in computing order) tail-free of inserted nodes, which
    # makes this hold.
    n_nodes = len(net.nodes)
    if net.links:
        last_link = max(net.links, key=lambda l: l.cmpseq)
        tail = nodes.get(last_link.nddsid)
        if tail is not None and tail.frmno != n_nodes:
            errors.append(
                f"link {last_link.lk_id} (last in computing order) ends at "
                f"node {tail.nd_id} with FRMNO {tail.frmno}, but the node "
                "count is "
                f"{n_nodes}; c1dlnk.f90 sizes the model's node array from "
                "this FRMNO (NumNodes = LNKEND(LK_COUNT))")

    # reach file: RC_NDUSID/RC_NDDSID must be a consistent DSID/USID pair
    for r in net.reaches:
        if r.ndusid not in nodes or r.nddsid not in nodes:
            errors.append(f"reach {r.rc_id}: endpoints "
                          f"{r.ndusid}/{r.nddsid} not nodes")
            continue
        u, d = nodes[r.ndusid], nodes[r.nddsid]
        if u.dsid != r.nddsid:
            errors.append(f"reach {r.rc_id}: node {r.ndusid} DSID={u.dsid} "
                          f"!= reach downstream node {r.nddsid}")
        if d.usid != r.ndusid:
            errors.append(f"reach {r.rc_id}: node {r.nddsid} USID={d.usid} "
                          f"!= reach upstream node {r.ndusid}")

    # channel file: a channel is a contiguous DSID *segment* from head to tail.
    # The tail's DSID normally continues into the next channel (via the
    # junction), so we only require that the tail is *reachable* from the head
    # by following DSID links -- not that the tail is terminal.
    for c in net.channels:
        if c.ndusid not in nodes or c.nddsid not in nodes:
            errors.append(f"channel {c.ch_id}: bad endpoint nodes")
            continue
        cur = c.ndusid
        ok = (c.ndusid == c.nddsid)
        for _ in range(len(nodes) + 1):
            if cur == c.nddsid:
                ok = True
                break
            n = nodes.get(cur)
            if n is None:
                errors.append(f"channel {c.ch_id}: broken DSID chain at {cur}")
                break
            if n.dsid in (NEGATIVE_ONE, n.nd_id):
                break
            cur = n.dsid
        if not ok:
            errors.append(f"channel {c.ch_id}: tail {c.nddsid} not reachable "
                          f"from head {c.ndusid} via DSID chain")

    # links
    for l in net.links:
        if l.lk_type not in (1, 2):
            errors.append(f"link {l.lk_id}: invalid LK_TYPE {l.lk_type}")
        if l.lk_type == 1 and l.lk_id not in channels:
            errors.append(f"link {l.lk_id}: LK_ID should be a CH_ID (type 1)")
        # A 1-point channel has no reach segments and carries the -1 sentinel
        # (the model ignores LK_RCUSID/LK_RCDSID anyway).
        if l.rcusid == NEGATIVE_ONE:
            if l.rcdsid != NEGATIVE_ONE:
                errors.append(f"link {l.lk_id}: LK_RCUSID=-1 requires "
                              "LK_RCDSID=-1 (1-point channel)")
        elif l.rcdsid == NEGATIVE_ONE:
            errors.append(f"link {l.lk_id}: LK_RCDSID=-1 requires "
                          "LK_RCUSID=-1 (1-point channel)")
        elif l.rcusid not in reaches or l.rcdsid not in reaches:
            errors.append(f"link {l.lk_id}: reach refs "
                          f"{l.rcusid}/{l.rcdsid} missing")

    # cross sections (geometry-level issues are tolerated by the model --
    # it never checks station ordering or row counts -- so they are
    # warnings, not errors)
    for cs in net.csecs:
        if not cs.points:
            warnings.append(f"cross section {cs.cs_id}: no profile points")
        else:
            ws = [p[1] for p in cs.points]
            if ws != sorted(ws):
                warnings.append(f"cross section {cs.cs_id}: CP_W not "
                                "increasing")

    if errors:
        raise NetworkValidationError(errors)
    return errors, warnings
