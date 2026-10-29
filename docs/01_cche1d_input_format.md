# CCHE1D channel-network input format

*Ground truth: the Fortran readers `cche1d_intelfortran/Flow/db_w_ntw_in.f90`,
`db_w_c1d_in.f90`, `db_w_cs_in.f90`; the GUI header
`cche1d_gui-vs2019/MyHeader.h`; and the known-good files in
`topagnps2cche1d/data_inputs/netchanTest_example_cche1d_input_files/` and
`topagnps2cche1d/documents/LoopTest/`.*

This is the authoritative description of what a **valid** CCHE1D channel network
looks like on disk and the invariants the Fortran import enforces. Everything
else in this folder (the builder, the TopAGNPS pipeline, the examples) is built
to produce input the model accepts.

## 1. What the model reads

The model imports a channel network from **six** tab-separated `.dat` files that
share one prefix (`<p>`). Every file has the same layout:

```
<record count>          <- line 1: number of data rows
COL1<TAB>COL2<TAB>...   <- line 2: header names
row 1
row 2
...
```

`-1` means "none"/absent. All ids are 1-based. The six files:

| File | Content | Columns |
|---|---|---|
| `<p>_nodes.dat` | all nodes + a trailing junction block | 12 per node |
| `<p>_reach.dat` | one row per node-to-node segment | 5 |
| `<p>_channel.dat` | one row per whole channel (head → tail) | 3 |
| `<p>_link.dat` | one row per link (= one channel) | 7 |
| `<p>_csec.dat` | one row per cross section | 6 |
| `<p>_csprf.dat` | one row per cross-section point | 7 |

These six are the only files the Fortran reader (`db_w_ntw_in.f90` et al.) reads
to build the network. **Two extra files the `topagnps2cche1d` library also
writes are auxiliary and are NOT read by the model:**

- `<p>_tw_cell.dat` — which cell drains into which network node (see
  [modifying-the-network](04_modifying_the_network.md)).
- `<p>_inflow_reaches.csv` — the removed reaches and the node each feeds
  (the coarsened inflow boundary conditions).

The Fortran import is **positional**: `READ(10,*) ND_ID(I), ND_FRMNO(I), ...` in
a `DO I = 1, ND_COUNT` loop, and **every node reference in the whole file set is
used as an array subscript** (`ND_DSID(I)`, `ND_USID(I)`, …). The hard invariant
is therefore that row `I` is node `ND_ID == I`.

## 2. Nodes (`<p>_nodes.dat`)

```
ND_ID  ND_FRMNO  ND_TYPE  ND_XC  ND_YC  ND_DSID  ND_USID  ND_US2ID  ND_CSID  ND_RSID  ND_STID  ND_SBID
```

(Each data row carries 12 values; the header line may name only 11 — the 12th,
trailing, is `ND_SBID`.)

- **ND_ID** — the file-row identity; must equal the row number (see §1).
- **ND_FRMNO** — the *computing id*: a permutation of `1..N` in
  upstream→downstream post-order (see §5). In the files this library writes,
  `ND_ID == ND_FRMNO`; that is a legal special case, and the model only requires
  `ND_FRMNO` to be a legal permutation.
- **ND_TYPE** — node role (table below, from `MyHeader.h`).
- **ND_XC / ND_YC** — planar (projected) coordinates, same CRS as the file.
- **ND_DSID** — downstream node id. **Never `-1`**: the outlet node uses the
  self-reference (`ND_DSID` = own `ND_ID`). A `-1` makes the import evaluate
  `NDCODE(-1)` out of bounds in `c1dlnk.f90` (only `FRMND` is allocated from
  −1; `NDCODE` is not).
- **ND_USID / ND_US2ID** — the first and second upstream-branch nodes (the tails
  of the upstream reaches that discharge into this node). `-1` = absent. For a
  two-branch junction these two columns are the whole story; branches 3–5 live
  in the junction block.
- **ND_CSID** — cross-section id for this node (`-1` = none). The model resolves
  it with a `DO CS / IF` that has **no else branch**, so a dangling or absent id
  is silently skipped (see §4).
- **ND_RSID** — routing/reach association; `-1` = none.
- **ND_STID / ND_SBID** — station / sub-basin bookkeeping (default 1).

### Node type codes (from `cche1d_gui-vs2019/MyHeader.h`, `NDTYPE_*`)

| Code | Name | Meaning |
|---|---|---|
| 0  | `SOURCE`       | source head (a channel with no upstream inflow, or whose upstream inflows were all removed) |
| 1  | `INFLOW`       | inflow boundary — exactly one upstream reach was removed and pours in here |
| 2  | `JUNCTION`     | confluence of 2–5 inflow branches |
| 3  | `ENDOFLINK`    | tail of a reach that joins a junction (sits at the junction head's XY) |
| 4  | `STRTUS`       | start of upstream (auto) |
| 5  | `AUTOADDED`    | auto added |
| 6  | `USERADDED`    | interior node (incl. a merged single-inflow confluence, see below) |
| 9  | `WATSHDOUTL`   | watershed outlet |
| 10 | `STRTDS`       | start of downstream (auto) |
| 11 | `BRIDGE`       | bridge |
| 12 | `CULVERT`      | culvert |
| 13 | `DROP`         | drop structure |
| 14 | `MEASFLUME`    | measurement flume |
| 15 | `DAM`          | dam |
| 19 | `UNKNOWN`      | unknown |
| 20 | `JUNCTSOURCE`  | junction source (loop head) |
| 21 | `JUNCTEND`     | junction end (loop tail) |

The `topagnps2cche1d` builder emits only the plain-routing types
`0, 1, 2, 3, 6, 9`. It does **not** place structure nodes (11–15) or loop nodes
(20/21) — see [modifying-the-network](04_modifying_the_network.md). The validator
accepts the full set `{0,1,2,3,4,5,6,9,10,11,12,13,14,15,19,20,21}`.

### Junction block (trailing, same file)

After the node rows:

```
ND_JUNCT
J  branch3  branch4  branch5     <- one row per type-2 node
```

- The row count equals the number of **type-2** nodes (one row per junction even
  when all extra branches are absent).
- Branches 1 and 2 come from the inline `ND_USID`/`ND_US2ID` of node `J`.
- `-1` in the file is clamped to `0` internally = "no branch".
- A junction may therefore have **2, 3, 4, or 5** inflow branches (CCHE1D's
  `ND_JUNCT5` limit). The builder rejects more than 5.
- Rows are stored **by `J` = the node's `ND_ID`** and read back by node-file row
  index, so block rows must be in `ND_ID` order (with the file invariant
  `ND_ID == row`, that is ascending `ND_ID`).
- The same block also carries **looped-channel** support: a loop channel is a
  channel between two junctions whose first node is type 20 (`JUNCTSOURCE`);
  `db_w_ntw_in.f90:296-305` normalises a type-20 branch into position 2 and
  `c1dlnk.f90:145-185` marks it negative in `ILNCHR` (the loop outflow). The
  builder never emits type 20/21 nodes — loops are GUI-only.

### Link-tail termination (the rule `c1dlnk.f90` enforces)

For every link the import computes the type of the node **downstream of the
link tail** (`NDCODE(NDDSND(LNKEND))`) and maps it to a link code:

| downstream type of the tail | link code |
|---|---|
| 2 (junction) | 3 — channel junction |
| 9 (outlet) | 2 — watershed outlet |
| 3 (ENDOFLINK) | 2 — **only if this is the last link in CMPSEQ** |
| 11 / 12 / 13 / 14 / 15 | 7 / 6 / 9 / (1 if last) 8 / 10 — structures |
| 10 (STRTDS) | re-read the tail's own type (a structure) |
| **anything else (0, 1, 6, …)** | **`C1DERRF` — the model stops at import** |

Consequence: **a reach whose tail points at a type-6 node is fatal input.**
That is why a confluence with a single kept inflow must **not** be written as
two channels (type-3 tail → type-6 head). The builder merges such a pair into
ONE continuous channel: the confluence pixel is a single interior type-6 node,
no `ENDOFLINK` tail, no branch, no junction-block row. (A 1-branch type-2
junction imports but makes `cs_adj_jnc.f90` adjust the thalweg of network
section 1 — a silent geometry corruption — so that is rejected too.)

Junction node count is not a surprise: a junction with `k` inflows has **k
branch nodes plus the type-2 head, all at the junction head's XY** — `k + 1`
overlapping nodes. Each branch node is either the inflow's own tail node
(when the tail pixel coincides with the head pixel) or a synthetic type-3
node the builder inserts at the head pixel (when the pixels differ — the
original tail keeps its coordinates and gains a reach segment to the branch
node; see [03_network_builder](03_network_builder.md)). The known-good
reference files show the shared-tail form (e.g. Goodwin junction 6: nodes 4, 5
type 3 + node 6 type 2, all at (5254.32, 6882.35)) — and the reference files
were hand-authored with tail pixels on the head pixel, so both forms are
legal; the builder emits the form the input geometry demands.

## 3. Reaches, channels, links

### `<p>_reach.dat`

```
RC_ID  RC_NDUSID  RC_NDDSID  RC_ORDER  RC_LENGTH
```

A "reach" here is **one segment between two consecutive nodes** (not a TopAGNPS
reach). A 10-node channel is 9 reaches. Each segment must be a real
`DSID`/`USID` link in the node table.

**`RC_ID` and `RC_ORDER` are informational.** The Fortran reader reads the reach
rows **by row position** (`DO RC = 1, RC_COUNT` uses `RC_NDUSID(RC)` /
`RC_LENGTH(RC)`); `RC_ID` is never used as an array index and is not
cross-referenced. Connectivity lives entirely in `RC_NDUSID`/`RC_NDDSID` and the
node table. Both a globally-unique `RC_ID` and a per-channel `RC_ID` convention
are valid input. This library writes globally-unique `RC_ID`s (1..N across the
file); the legacy reference files in `data_outputs/` use a per-channel convention.

### `<p>_channel.dat`

```
CH_ID  CH_NDUSID  CH_NDDSID
```

One row per channel: its head and tail node ids. A channel is a contiguous
`DSID` segment chain from head to tail; the tail's `DSID` continues into the
next channel (through the junction node). A **1-point channel** (head == tail)
is legal — it comes from an unexpanded boundary single-node reach.

### `<p>_link.dat`

```
LK_ID  LK_CMPSEQ  LK_NDUSID  LK_NDDSID  LK_RCUSID  LK_RCDSID  LK_TYPE
```

One link per channel. For `LK_TYPE = 1` (channel), `LK_ID == CH_ID`.
`LK_RCUSID`/`LK_RCDSID` = the channel's first/last `RC_ID`; the model reads these
but never uses them. A **1-point channel** has zero reach segments and carries
`LK_RCUSID = LK_RCDSID = -1`. `LK_TYPE = 2` is reserved for hydraulic structures.
In the files this library writes, `CH_ID == LK_ID == LK_CMPSEQ` (the
compute-order position) — a legal special case the model tolerates.

## 4. Cross sections

### `<p>_csec.dat`

```
CS_ID  CS_NPTS  CS_LOB  CS_LBT  CS_ROB  CS_RBT  [CS_SVTYPE  CS_TYPE  CS_ORIGIN  CS_STATION]
```

The Fortran reads the first six values positionally. `CS_ID` = the node that owns
the section (one section per node; the writer sets `ND_CSID == CS_ID == ND_ID`).
`CS_LOB`/`CS_LBT`/`CS_ROB`/`CS_RBT` are the four **1-based point indices** marking
the left-floodplain / left-bank-toe / right-bank-toe / right-floodplain
transitions. The Fortran swaps its internal `CS_ROB`/`CS_RBT` variables vs the
file order — but identically for every file it reads — so writing in **header
order** is correct.

### `<p>_csprf.dat`

```
CP_ID  CP_CSID  CP_POSIDX  CP_W  CP_Z  CP_RGH  CP_BLOCK
```

One row per profile point; `CP_CSID` ties it to its section, `CP_POSIDX` is the
1-based position within the section, `CP_W` is the transverse station (left →
right), `CP_Z` the bed elevation, `CP_RGH` Manning's n, `CP_BLOCK` 1 open /
0 closed.

**Cross sections are optional at import.** The model resolves each node's
`ND_CSID` with `DO CS = 1, CS_COUNT / IF(CS_ID(CS) .EQ. NDCSID) … ENDIF ENDDO`
— **no else branch** — so an unmatched id (or an empty `_csec.dat`) is silently
skipped. Real example: the `documents/LoopTest` ground-truth files carry
`ND_CSID = 1..12` on all 12 nodes but ship a **0-record** `_csec.dat`, and the
model imports them. The import performs **no geometry check** at all on these
rows (station ordering, row counts); out-of-order or missing points import
without complaint. The library's validator reports those as **warnings**, not
errors.

## 5. Computing order (the only ordering rule)

`ND_FRMNO` must be a **permutation of `1..N`** produced by a post-order DFS from
the outlet, following each node's upstream branches, visiting predecessor
**channels in descending id order** (equivalently
`tools.custom_dfs_traversal_sorted_predecessors(visit_descending_order=True,
postorder=True)`; the builder's `network.cche1d_compute_order` is the same
algorithm). Within each channel the node chain is numbered **head → tail**. The
consequence the solver relies on:

> **a junction's inflow branches must all have a smaller `ND_FRMNO` than the
> junction node itself** — i.e. water is computed upstream before it is routed
> downstream.

This is the invariant `network.validate_network` checks.

## 6. What the solver actually uses

Verified against the model core (outside the readers): `RC_ID`, `CH_ID`, `LK_ID`,
`LK_CMPSEQ` are allocated but **not referenced by the solver's math** — they are
bookkeeping identifiers for display, post-processing and file consistency. The
solver uses:

- **positional indexing** (array slot `I`),
- **`ND_DSID / ND_USID / ND_US2ID / ND_JUNCT5`** for connectivity,
- **`RC_NDUSID / RC_NDDSID`** to attach each reach to its two nodes,
- **`ND_FRMNO`** as the state-solver iteration order.

So a network is valid iff: (a) row `I` is node `ND_ID == I`; (b) `ND_FRMNO` is a
legal post-order permutation; (c) every type-2 node has a junction-block row and
its branches 3–5 resolve; (d) every reach attaches to a consistent `DSID`/`USID`
pair; (e) every channel/link is a connected `DSID` chain; (f) cross-section
points are (conventionally) sorted by increasing `CP_W`.

## 7. GUI → shapefiles

The Windows GUI does not read the shapefiles for the model — it bundles Shapelib
and, in `FileManagerCCHE1D.cpp`, writes the **same six tables** as shapefiles
(`_nodes.shp` point, `_reach.shp`/`_chann.shp`/`_link.shp` arc, `_csec.shp` /
`_csprf.shp` per-section) with the **same prefix** and the same DBF columns. The
shapefiles are a lossless visual mirror for display/editing; the `.dat` files
remain the single source of truth for the model.

## 8. Checklist for a hand-written network

1. Number nodes 1..N; keep row = `ND_ID`.
2. Fill `ND_DSID` (self-reference at the outlet; never −1 elsewhere),
   `ND_USID`/`ND_US2ID` for junctions
   (branch 1 = tail of the smallest-id inflow, branch 2 = next — the generator's
   convention).
3. Type every confluence with ≥2 inflows as `2`; multi-inflow junction tails as
   `3`; a lone removed inflow as `1`; a head with no kept inflow as `0`; the
   outlet as `9`; interior nodes as `6`. A single-inflow confluence is **one**
   type-6 interior node (merged), never a `3`→`6` pair.
4. Emit one junction row per type-2 node, padding branches 3–5 with `-1`.
5. Compute `ND_FRMNO` by the post-order DFS of §5.
6. Emit one reach per node-pair inside each channel.
7. Emit one channel and one link row per channel.
8. Give every node a CS (or `ND_CSID = -1`); keep `CP_W` increasing.
9. Validate with `topagnps2cche1d.network.validate_network`.
