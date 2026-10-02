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
  Branch 1 (USID) = tail node of the largest-id inflow channel, branch 2
  (US2ID) = the next, matching the two-inflow rule in the reference code.
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

class ChannelNetworkBuilder:
    """
    Build a CCHE1D network from a manual topology definition.

    Parameters
    ----------
    channels : dict {reach_id: (receiving_reach_id_or_None, [(x, y), ...])}
        Each entry defines one channel by the id of the reach it discharges
        into (``None`` for the outlet channel) and an ordered list of
        (x, y) points from its upstream (head) end to its downstream (tail)
        end.  The reach ids should be the *TopAGNPS-style* ids (ascending
        from upstream to downstream), as in the original import; the builder
        re-derives the CCHE1D channel/compute order from them.
    strahler : dict {reach_id: strahler_number}, optional (default 1).
    resample_step : float or None
        If given, interior nodes are sampled along each channel geometry at
        this spacing (in coordinate units).
    cs_template : (lob, lbt, rob, rbt) or None
        If given, every non-junction, non-outlet node gets a cross section
        from this template with a single bed point.
    head_type : int, default NODE_SOURCE
        ND_TYPE for the head node of a channel with no upstream inflows.
    """

    def __init__(self, channels, strahler=None, resample_step=None,
                 cs_template=None, head_type=NODE_SOURCE):
        if not channels:
            raise ValueError("channels dict is empty")
        self.channel_defs = dict(channels)
        self.strahler = strahler or {}
        self.resample_step = resample_step
        self.cs_template = cs_template
        self.head_type = head_type

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
        npts = {r: len(geom[r]) for r in raw_ids}

        # ---- number nodes head->tail in computing order; ND_ID == FRMNO ----
        frmno = {}
        counter = 0
        for r in compute_order:
            for k in range(npts[r]):
                counter += 1
                frmno[(r, k)] = counter
        if counter != sum(npts.values()):
            raise RuntimeError("node count mismatch (unreachable channel?)")

        nodes = {}
        for (r, k), nd_id in frmno.items():
            x, y = geom[r][k]
            nodes[nd_id] = NodeSpec(
                nd_id=nd_id, frmno=nd_id, nd_type=NODE_USERADDED,
                xc=x, yc=y, dsid=NEGATIVE_ONE, usid=NEGATIVE_ONE,
                us2id=NEGATIVE_ONE)

        # intra-channel links + tail node type
        for r in raw_ids:
            n = npts[r]
            head = frmno[(r, 0)]
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
            else:
                # channel tail -> head node of the receiving channel
                nodes[tail].nd_type = NODE_END_LINK
                nodes[tail].dsid = frmno[(recv, 0)]

        # head node type + junction wiring
        junctions = []
        for r in raw_ids:
            inflows = [c for c in raw_ids if defs[c][0] == r]
            head = frmno[(r, 0)]
            if not inflows:
                # source channel: no upstream inflow
                nodes[head].nd_type = self.head_type
                nodes[head].usid = NEGATIVE_ONE
                continue
            # a channel head with >=1 inflow: junction if >=2, trivial if 1
            # branch order: inflows by descending raw reach id (== descending
            # TopAGNPS id), the highest first -> USID, then US2ID, then 3..5
            ends = [frmno[(c, npts[c] - 1)] for c in
                    sorted(inflows, key=lambda c: -c)]
            if len(ends) > MAX_JUNCTION_BRANCHES:
                raise ValueError(
                    f"junction node {head} has {len(ends)} inflows; "
                    f"CCHE1D supports at most {MAX_JUNCTION_BRANCHES}")
            for e in ends:
                nodes[e].dsid = head
            if len(ends) == 1:
                # trivial connection: head stays type 6, single USID
                nodes[head].nd_type = NODE_USERADDED
                nodes[head].usid = ends[0]
                nodes[head].us2id = NEGATIVE_ONE
            else:
                # proper junction
                nodes[head].nd_type = NODE_JUNCTION
                nodes[head].usid = ends[0]
                nodes[head].us2id = ends[1]
                # every type-2 node gets a junction-block row; branches 3..5
                # beyond the inflows are padded with -1 (Fortran MAX(x,0))
                junctions.append(JunctionExtra(
                    head, tuple((ends[2:] + [-1, -1, -1])[:3])))

        # ---- channels, reaches, links (in computing order) ----------------
        channels, reaches, links = [], [], []
        rc_id = 0
        for r in compute_order:
            n = npts[r]
            head, tail = frmno[(r, 0)], frmno[(r, n - 1)]
            channels.append(ChannelSpec(ch_id[r], head, tail))
            xs = [geom[r][k][0] for k in range(n)]
            ys = [geom[r][k][1] for k in range(n)]
            first_rc = rc_id + 1
            for k in range(n - 1):
                rc_id += 1
                a, b = frmno[(r, k)], frmno[(r, k + 1)]
                length = math.hypot(ys[k + 1] - ys[k], xs[k + 1] - xs[k])
                reaches.append(ReachSpec(rc_id, a, b,
                                         self.strahler.get(r, 1), length))
            links.append(LinkSpec(ch_id[r], ch_id[r], head, tail,
                                  first_rc, rc_id, 1))

        # ---- cross sections ------------------------------------------------
        csecs = []
        if self.cs_template is not None:
            lob, lbt, rob, rbt = self.cs_template
            for nd in nodes.values():
                if nd.nd_type in (NODE_JUNCTION, NODE_WATSHDOUTL):
                    continue  # junction / outlet nodes carry no cross section
                nd.csid = nd.nd_id
                csecs.append(CrossSectionSpec(
                    cs_id=nd.nd_id, lob=lob, lbt=lbt, rob=rob, rbt=rbt,
                    points=[(1, 0.0, 0.0, 0.03, 1)]))

        return Cche1dNetwork(
            nodes=sorted(nodes.values(), key=lambda n: n.frmno),
            reaches=reaches, channels=channels, links=links,
            csecs=csecs, junctions=junctions,
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
    # one junction-block row per type-2 node (branch order 3,4,5; -1 = absent)
    junc_nodes = sorted((j for j in net.junctions),
                        key=lambda j: next(m.frmno for m in net.nodes
                                           if m.nd_id == j.node_id))
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
                            _fmt_float(z), f"{rgh:.2f}", block))
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
    (db_w_ntw_in.f90 / db_w_c1d_in.f90 / db_w_cs_in.f90).  Returns [] if the
    network is valid, otherwise raises NetworkValidationError with the list
    of problems.
    """
    errors = []
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
            errors.append(f"node {n.nd_id}: ND_CSID={n.csid} not in csec file")

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
        if l.rcusid not in reaches or l.rcdsid not in reaches:
            errors.append(f"link {l.lk_id}: reach refs "
                          f"{l.rcusid}/{l.rcdsid} missing")

    # cross sections
    for cs in net.csecs:
        if not cs.points:
            errors.append(f"cross section {cs.cs_id}: no profile points")
        else:
            ws = [p[1] for p in cs.points]
            if ws != sorted(ws):
                errors.append(f"cross section {cs.cs_id}: CP_W not increasing")

    if errors:
        raise NetworkValidationError(errors)
    return []
