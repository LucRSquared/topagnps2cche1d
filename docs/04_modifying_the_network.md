# Modifying the network

Coarsening, resampling, cross sections, and the removed-reach (inflow) record —
the operations you apply to a built network, in either the manual path or the
TopAGNPS path.

## Strahler coarsening

Strahler numbers are assigned from the **full** connectivity by default
(`assign_strahler_number_to_reaches(mode="fully_connected")`), so coarsening is
stable: a reach's order never exceeds its receiver's. Two mechanisms compose
(either, both, or neither):

1. **The `ignore` flag** — the generic path.
   `w.ignore_reaches_with_strahler_leq(t)` sets `reach.ignore` on every reach
   with order ≤ `t` (and on the upstream subtrees they feed). The `Watershed`
   re-derives its `current_graph` and the CCHE1D build works on the kept
   sub-network.
2. **`strahler_drop`** — the builder path.
   `build_network_from_reaches(reaches, strahler_drop=t)` drops order ≤ `t`
   inside `network.py` without touching any `Watershed` state. Used when
   building directly from `Reach` objects (a manual network).

Because order is monotone downstream, the kept set is always a connected
sub-network with exactly one outlet — no connectivity repair is needed.

### Head-node classification after coarsening

For each kept reach, its head node type is derived from the upstream reaches in
the *full* set (the full rule is in [03_network_builder](03_network_builder.md)):

| Upstream reaches (full set) | Head type | Meaning |
|---|---|---|
| ≥ 2 kept, or none at all | junction (2) / source (0) | confluence or true source |
| exactly 1 kept | merged channel (interior 6) | single inflow; the pair is one continuous channel |
| exactly 1 removed, ≥1 kept | junction/trivial (2/6) | removed reach recorded, **not** a BC head |
| exactly 1 removed, none kept, 1-node link | **inflow boundary (1)** on the head | a type-1 head is legal only for a 1-node link (c1dlnk.f90:157-158) |
| exactly 1 removed, none kept, multi-node link | **source (0)** head + type-1 label one node downstream | a type-1 multi-node head stops the import (C1DERRF) |
| ≥ 2 removed, none kept | **source (0)** | several removed reaches |

A head that keeps any inflow stays a junction or a merged confluence; the
removed reach is only listed in the inflow record. Type-1/type-0 heads appear
only when *all* of a head's upstream reaches are removed, and the type-1
label itself sits on the head only for a 1-node link — otherwise one node
downstream of the (type-0) head. The label is display-only; the model never
reads it, and the inflow record always points at the head.

## The removed-reach (inflow) record

Coarsening is lossy: water that flowed through the removed reaches now enters the
kept network at the heads of the kept reaches. CCHE1D needs these as inflow
boundary conditions, so the library records **which reaches were removed and
which node each one feeds**. This record now lives in the **network layer**
(`topagnps2cche1d.network`) so it is produced for a hand-built (manual) network
as well as a TopAGNPS import:

- `removed_reach_inflow_pairs(reaches, strahler_drop)` →
  `[(removed_id, receiving_reach_id)]` — the generic-reach form.
- `inflow_reaches_records(net)` → `[(removed_id, receiving ND_ID)]` — the
  node-id form, using `net.channel_heads` (the head node of the kept channel).
- `write_inflow_reaches_csv(net, path)` → writes the CSV
  (`inflow_topagnps_reach_id,receiving_node_id`).
- `Watershed.get_list_of_inflow_reaches_df()` returns the same record as a
  DataFrame, delegating to `inflow_reaches_records` — so the manual and TopAGNPS
  paths share one implementation and can never disagree.

The receiving node is the **head** of the kept reach — the inflow boundary node
the builder classifies — so the stamp, the builder's head typing, and the
model-facing record all point at the same node. (The legacy pipeline stamped the
*second* node of a channel; that has been unified to the head.)

### What to do with the inflow file

For each row in `*_inflow_reaches.csv`, run/generate the matching AnnAGNPS reach
output and wire it into CCHE1D as a time/instantaneous inflow at the given node
(`ND_ID = receiving_node_id`). Without this, the coarsened model loses that
upstream runoff.

### Manual path, end to end

```python
from topagnps2cche1d.network import (
    build_network_from_reaches, validate_network,
    write_cche1d_dat_files, write_inflow_reaches_csv,
)

# `reaches` is an iterable of generic Reach objects you built by hand.
reaches = [ ... ]                # build your topology (see 03 + the quickstart)

# Coarsen by Strahler order (and/or set reach.ignore on specific reaches).
net = build_network_from_reaches(reaches, strahler_drop=1)
errors, _ = validate_network(net)
assert not errors

write_cche1d_dat_files(net, "out", prefix="mynetwork")
write_inflow_reaches_csv(net, "out/mynetwork_inflow_reaches.csv")  # removed reaches
```

`tests/test_manual_inflow.py` exercises this path (no raster, no TopAGNPS) and
pins the head-node receiving semantics.

## Resampling

- `w.resample_reaches(id_list, step=...)` or `numpoints=` re-spaces every node
  along each reach geometry (US/DS nodes and their types are preserved); the
  cached CCHE1D network is invalidated automatically.
- The build can also resample on the fly via
  `build_cche1d_network(resample_step=...)` / `ChannelNetworkBuilder(resample_step=...)`
  (builder-side `_interpolate`), which leaves the generic node spacing untouched.

## Cross sections

- `w.assign_cross_section_to_all_points(**kwargs)` — assigns a `CrossSection` to
  every node of the kept reaches (trapezoid-with-floodplain default; arbitrary
  user-defined sections via `cs_type="user_defined", ws=..., zs=...,
  lfp_idx=..., ...`).
- `w.assign_cross_section_to_node_byid(node_id, **kwargs)` — per-node override.
- `w.adjust_cross_sections_elevation_with_slope(outlet_elevation=0)` — shifts
  each section's thalweg so elevations follow the channel slope from the outlet
  up (walks the US→DS node chain; the outlet node keeps the reference elevation).
- The build maps each CCHE1D node back to its generic node (provenance) and
  assigns that node's section; nodes without one fall back to the optional
  `cs_template` argument.

Arbitrary (non-trapezoidal) sections are first-class: any `ws`/`zs` arrays with
the four 1-indexed indices (LFP/LBT/RBT/RFP) round-trip through csec/csprf.
`cross_section_to_spec(cs)` is the direct converter from a `CrossSection` to a
`CrossSectionSpec` for the builder.

## Hydraulic structures (node types 11–15)

The constants exist in `network.py` (`NODE_BRIDGE`, `NODE_CULVERT`,
`NODE_DROP`, `NODE_MEASFLUME`, `NODE_DAM`) and the validator accepts them as node
types. The **builder does not place structure links** (`LK_TYPE = 2`): it emits
channel links only. Adding a structure requires post-processing the built
`Cche1dNetwork` (swap the node `ND_TYPE`, add the `LK_TYPE 2` link, add the
corresponding structure-table row) or a future builder extension — tracked as a
task, not silently emulated.
