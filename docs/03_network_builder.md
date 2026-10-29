# Network builder — `topagnps2cche1d.network`

*Stdlib-only: `network.py` imports nothing beyond the Python standard library,
so it runs in any environment — no pandas, numpy, networkx, or GDAL required.
Run the tests with bare `python3 tests/test_network.py` or under pixi.*

This module is the single source of truth for the CCHE1D encoding. Given a reach
topology — declared by hand or read from any source — it produces a complete,
Fortran-readable CCHE1D network and validates it. It is the encoder both the
manual path and the TopAGNPS path go through.

## 1. The data model

`Cche1dNetwork` holds the six tables plus the bookkeeping maps:

- `nodes` (list of `NodeSpec`) — 12 columns per node.
- `reaches` (list of `ReachSpec`) — one row per node-to-node segment.
- `channels` (list of `ChannelSpec`) — one row per whole channel.
- `links` (list of `LinkSpec`) — one row per link.
- `csecs` (list of `CrossSectionSpec`) — one section per node that has one,
  with its `points` (posidx, w, z, rgh, block).
- `junctions` (list of `JunctionExtra`) — branches 3–5 for each type-2 node.
- `channel_heads` — `{generic reach id: head ND_ID}` for every built channel.
- `node_origin` / `node_provenance` — `{ND_ID: generic node id}` maps (see below).
- `removed_reaches` — `[(removed_reach_id, receiving_reach_id)]` for reaches
  dropped by coarsening (populated by `build_network_from_reaches`).

`node_origin` is the **bijective** map from a network node to the generic node
at that channel's own point; `node_provenance` is a variant where a type-3
branch node standing at a junction head pixel (either the shared tail node or
a synthetic inserted one) points at the *receiving* head's generic node (so it
carries the cross-section profile standing at that pixel).
`node_origin` never lists synthetic inserted nodes. Cross-section assignment
uses `node_provenance`; per-node bookkeeping (cell drainage) uses
`node_origin`.

## 2. Defining a network

The builder accepts either of two input forms:

1. **A raw dict** `{reach_id: (receiving_reach_id_or_None, [(x, y), …])}` —
   each entry is one channel, points ordered upstream (head) → downstream
   (tail). The outlet channel has `receiving = None`.
2. **An iterable of generic `Reach` objects** (the preferred form — the same
   classes the TopAGNPS path uses). `ChannelNetworkBuilder` detects a
   non-dict iterable and converts it via `channel_defs_from_reaches`.

```python
from topagnps2cche1d.network import (
    ChannelNetworkBuilder, write_cche1d_dat_files,
    read_cche1d_network, validate_network,
)

channels = {
    1: (None, [(0, 0), (5, 0), (10, 0)]),        # outlet
    2: (1,    [(20, -20), (15, -10), (10, 0)]),  # left inflow  -> ch1 head
    3: (1,    [(20, 20), (15, 10), (10, 0)]),    # right inflow -> ch1 head
}
net = ChannelNetworkBuilder(channels,
                            resample_step=4.0,        # node spacing along reaches
                            cs_template=(2, 2, 2, 2)) # 4-tuple: single bed point
net.build()
write_cche1d_dat_files(net, "out_dir", prefix="mynetwork")
```

`channel_defs_from_reaches(reaches)` is the explicit conversion function
(duck-typed: anything with `.id`, `.receiving_reach_id`, `.us_nd_id`, `.nodes`
and node `.x/.y/.id/.dsid` works). Nodes are read US→DS by walking the `dsid`
chain. A reach whose `receiving_reach_id` is `None` **or not present among the
provided reaches** (TopAGNPS marks the outlet with a dangling id) is treated as
the outlet. `tests/test_reach_adapter.py` verifies a network built from real
`Reach` objects is node-for-node identical to one built from the equivalent dict.

### `build_network_from_reaches` — the high-level entry point

This is the function the TopAGNPS path and any coarsening workflow use. It adds
Strahler coarsening and head-node classification on top of the builder:

```python
from topagnps2cche1d.network import build_network_from_reaches
net = build_network_from_reaches(reaches, strahler_drop=0,
                                 resample_step=None, cs_template=None,
                                 cs_by_generic_id=None)
```

- **Keep rule** (shared with the inflow record via `_kept_ids`): a reach is kept
  when it is not `ignore` and its Strahler order is above `strahler_drop`. A
  reach with no Strahler order (a manual network) is kept unless ignored.
- **Head-node classification** of each kept reach, from the upstream reaches in
  the *full* set:

  | Upstream reaches (full set) | Head type | Meaning |
  |---|---|---|
  | ≥ 2 kept, or none at all | junction (2) / source (0) | normal confluence or true source |
  | exactly 1 kept | trivial connection (6) | single inflow, merged channel |
  | exactly 1 removed, ≥1 kept | junction/trivial (2/6) | removed reach only *recorded*; the head is never a BC boundary |
  | exactly 1 removed, none kept, 1-node link | **inflow boundary (1)** on the head itself | a type-1 head is legal only for a 1-node link (c1dlnk.f90:157-158) |
  | exactly 1 removed, none kept, multi-node link | **source (0)** head + type-1 label one node downstream | a type-1 head of a multi-node link stops the model at import (C1DERRF) |
  | ≥ 2 removed, none kept | **source (0)** | several removed reaches; the record points at the head |

  A head keeps a kept inflow ⇒ it is a junction or trivial connection, **not**
  a type-1 boundary; the removed reach is only recorded in the inflow list.
  Type-1 / type-0 heads appear only when *all* upstream reaches of a head are
  removed. Wherever a type-1 label is placed (head of a 1-node link, or the
  node just downstream of a multi-node boundary head), a functional type
  already assigned to that node (3 branch / 9 outlet) wins; the label is
  display-only — the model never reads it, and the removed-reach inflow record
  always points at the head node.

`build_network_from_reaches` also sets `net.removed_reaches` (via
`removed_reach_inflow_pairs`) so the caller can emit the removed-reach inflow
record without re-deriving it — this is what makes a hand-built network keep its
inflow bookkeeping (see §4).

## 3. What `build()` does

1. **Resamples** each channel polyline at `resample_step` (constant spacing,
   endpoints kept) when given.
2. **Computes the computing order** — post-order DFS from the outlet over the
   reach graph, predecessor channels visited in descending id
   (`cche1d_compute_order`, identical to
   `tools.custom_dfs_traversal_sorted_predecessors`). `CH_ID` = position in that
   traversal.
3. **Wires junctions and confluences.** The governing invariant (verified
   against the `netchan` and `LoopTest` reference files and the GUI/fortran
   sources): **no node that belongs to a geometrically defined reach ever
   changes coordinates**. Junctions are handled by *adding* nodes:

   - A channel head with ≥2 inflow tails becomes a **type-2 node** with a
     junction-block row. Branch order: inflows sorted by **ascending** reach
     id → branch 1 (`ND_USID`), branch 2 (`ND_US2ID`), then 3–5 in the block
     (`-1` padding). The builder rejects more than 5 inflows (CCHE1D's
     `ND_JUNCT5` limit).
   - Each inflow's **branch node** is the tributary's tail node when the tail
     pixel coincides with the junction head pixel (shared tail — the tail
     becomes type 3 in place). When the pixels differ, the builder **inserts
     one synthetic type-3 node** at the junction head pixel: the original tail
     keeps its coordinates and its interior (type-6) type, its `DSID` points
     at the new branch node, and a reach segment (tail → branch) is added.
     The channel's tail is the branch node in both cases (netchan:
     `CH_NDDSID` is the type-3 node standing at the junction pixel).
   - A head with exactly one kept inflow is a **single-inflow confluence**:
     the pair merges into one continuous channel (splitting it would create
     a link ending at an interior type-6 node, which `c1dlnk.f90` cannot
     code — `C1DERRF`). Shared pixel: the single confluence node is type 6.
     Distinct pixels: both nodes are kept at their own coordinates and joined
     by a cross-boundary reach segment. No ENDOFLINK tail, no junction-block
     row.
   - The outlet channel's tail node is type 9 with self-`DSID`.
4. **Types boundary heads**: a head with no kept inflow takes its type from
   the `head_types` override (see §2) — including the 1-node-link vs
   multi-node-link rule for where the type-1 inflow-boundary label sits.
5. **Numbers nodes** head→tail in computing order (`ND_ID == ND_FRMNO`), and
   assigns the outlet node type 9.
6. **Numbers reaches** (globally-unique `RC_ID` in this library) and channels /
   links (`LK_ID == CH_ID`).
7. **Generates cross sections** for every node: a per-node `CrossSection` looked
   up by provenance (`cs_by_generic_id`), else the shared `cs_template` (a
   4-tuple yields a single bed point; a full `CrossSection` object is applied to
   every node). With neither, the network has no cross sections.

### 3-way (and up to 5-way) confluences

Declare three or more channels into the same head; the extra branches go into
`JunctionExtra.branches345` and the junction block. Verified by
`test_builder_three_way_junction`. Loops/divergences (node types 20/21, signed
reach directions) are **not** expressed by the builder today — it is a
tree-of-channels; a loop must be post-processed on the produced files.

## 4. The removed-reach inflow record (manual + TopAGNPS)

Coarsening is lossy: water that flowed through removed reaches now enters the
kept network at the heads of the kept reaches. CCHE1D needs these as inflow
boundary conditions, so the library records **which reaches were removed and
which node each one feeds**. This now lives in the network layer so it works for
a hand-built network too:

- `removed_reach_inflow_pairs(reaches, strahler_drop)` →
  `[(removed_id, receiving_id)]` — removed reaches that are *direct* upstream
  neighbours of a kept reach (a removed reach whose receiver is also removed is
  absorbed into that receiver's record).
- `build_network_from_reaches` stores these on `net.removed_reaches` and
  `build()` stores `net.channel_heads` (`{reach id: head ND_ID}`).
- `inflow_reaches_records(net)` → `[(removed_id, receiving ND_ID)]` — the
  node-id form, using `channel_heads`.
- `write_inflow_reaches_csv(net, path)` → writes
  `inflow_topagnps_reach_id,receiving_node_id` (the same schema
  `Watershed.get_list_of_inflow_reaches_df` produces).
- `Watershed.get_list_of_inflow_reaches_df` delegates to
  `inflow_reaches_records`, so the manual and TopAGNPS paths can never
  disagree.

A pure manual build with no ignored reaches returns an empty record (the CSV
still carries its header).

## 5. Writer / reader

- `write_cche1d_dat_files(net, out_dir, prefix)` → the six core files. Node rows
  are written **in ND_ID order with all 12 columns** (the Fortran reader is
  positional), followed by the junction block (one row per type-2 node,
  `J b3 b4 b5`, `-1` padding).
- `read_cche1d_network(prefix_dir)` → a `Cche1dNetwork` with the same
  positional semantics as the Fortran reader (also `read_cche1d_nodes_dat` for a
  single file). The prefix is inferred from the `*_nodes.dat` filename.
- Round-trip is **lossless** on the ground-truth Goodwin-Creek files and on
  builder output (test-verified).

## 6. The validator (reverse-engineered from the Fortran import)

`validate_network(net) -> (errors, warnings)` and `NetworkValidationError`
(raised when `errors` is non-empty, with the full list attached). **Errors** are
structural — the Fortran import would stop or produce a broken model. **Warnings**
mirror things the import *tolerates*; a warnings-only result means "the model
imports this; the cross-section geometry may be incomplete".

Error-level checks, in the order the model needs them:

| # | Check |
|---|---|
| 1 | no duplicate `ND_ID`/`CH_ID`; ids start at 1, count matches |
| 2 | every `ND_DSID/USID/US2ID` resolves to an existing node (or −1) |
| 3 | `ND_FRMNO` is a permutation of 1..N |
| 4 | **inflow-before-junction**: `FRMNO(branch) < FRMNO(junction)` for every inflow branch |
| 5 | type-2 nodes have 2–5 inflow branches; each has a junction-block row (and vice versa); branches 3–5 resolve; no duplicate branch; each branch's `DSID` points back at the junction |
| 6 | every reach attaches to a consistent `DSID`/`USID` node pair |
| 7 | each channel is a connected `DSID` chain head→tail; link `RC` refs resolve (a 1-point channel carries the `-1` sentinel on both) |
| 8 | **link-tail termination** (c1dlnk.f90:202-278): the node downstream of a link's tail must be a junction (2), outlet (9), or a hydraulic-structure node (11–15) — or an end-of-link (3) node on the *last* link. Anything else (a type-6 interior node, a split channel at a confluence, …) has no link code in `c1dlnk.f90` → `C1DERRF` |
| 9 | **head-side rule** (c1dlnk.f90:125-163): a multi-node link's head must be a source (0), junction (2), STRTDS (10) or JUNCTSOURCE (20) node; any other type — in particular a **type-1 inflow-boundary node** — leaves `ILNCHR(L,3)` unassigned → `C1DERRF`. A 1-node link may start at any type (that is the only legal home for a type-1 head) |
| 10 | **`FRMNO`-N rule** (c1dlnk.f90:296): the model sizes its node array from the last link's tail FRMNO (`NumNodes = LNKEND(LK_COUNT)`), so the last link (by computing order) must end at the node with `FRMNO = N` |

Warning-level (import-tolerated): unmatched `ND_CSID` (the model's `DO CS / IF`
has no else branch — e.g. the `documents/LoopTest` files pair 12 nodes with
`ND_CSID=1..12` but an **empty** `_csec.dat` and still import), a CS without
profile points, non-increasing `CP_W`. Negative tests prove the error level
fires (broken `DSID`, inflow computed after its junction, a split channel
ending downstream of a type-6 node, a type-1 multi-node head, a wrong last
link tail `FRMNO`); the LoopTest test proves the warning level does not
over-reject.

## 7. Running the tests

```bash
cd ~/projects/topagnps2cche1d
pixi install            # first time only
pixi run -e test test   # fast suite: network units + ground-truth fixtures +
                        # topology/bookkeeping + manual parabolic + manual
                        # removed-reach inflow + full Ohio e2e (seconds)
python3 tests/test_network.py   # bare-interpreter fallback (stdlib only)
```

## 8. Open items

- **Loop/divergence builder support** — express split-then-merge topologies
  declaratively (node types 20/21, signed reach directions).
- **Hydraulic structures** — place `LK_TYPE 2` links and structure nodes
  (11–15); the builder emits channel links only today (see
  [04_modifying_the_network](04_modifying_the_network.md)).
