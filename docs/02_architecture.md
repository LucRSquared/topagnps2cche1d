# Architecture and module scope

*Repo: `~/projects/topagnps2cche1d` (branch `cche1d-consolidation`), Python 3.10+,
environment managed by **pixi**. This is the single place that explains what each
module does and where the responsibility boundaries are.*

## The two layers and one boundary

```
TopAGNPS output                CCHE1D input
(3 files)                      (6 core files + 2 auxiliary)
     |                                ^
     v                                |
+----------------------+              |
| Watershed            |  generic     |
|  (watershed.py)      |  reaches     |
|  - import CSV/ASC    | ------------>+---- network.py ----+
|  - geometry, orient  | (Reach/Node) |   build, validate, |
|  - strahler coarsen  |              |   write, read      |
|  - inflow bookkeep   |              |   (manual + imported)
+----------------------+              |
                |                     |
                +--- frozen Reach (reach.py) / Node (node.py) ---+
```

- **`watershed.py` — the TopAGNPS layer.** Owns importing the TopAGNPS CSVs and
  the reach-ID raster, building the generic reach/cell topology, node geometry
  and US/DS orientation, Strahler order, the `ignore`-flag coarsening,
  resampling, cross sections, and the cell/removed-reach inflow bookkeeping. It
  does **not** own CCHE1D numbering; it delegates the CCHE1D build to
  `network.py`.
- **`network.py` — the CCHE1D layer.** Owns the channel-network architecture:
  the `Cche1dNetwork` data model, the builder (`build_network_from_reaches` /
  `ChannelNetworkBuilder`), the validator (`validate_network`), the writers
  (`write_cche1d_dat_files`, `write_inflow_reaches_csv`) and the reader
  (`read_cche1d_network`). It has **no import of `watershed`** and works with any
  iterable of `Reach`-like objects — a hand-built (manual) network and an
  imported one are indistinguishable to the Fortran model.
- **`reach.py` / `node.py` — the frozen shared contract.** Both layers read
  `.id`, `.receiving_reach_id`, `.strahler_number`, `.ignore`, `.us_nd_id`,
  `.ds_nd_id`, `.nodes`, and node `.x/.y/.id/.usid/.dsid/.csid`. These classes
  must not be modified (see the standing rule on `Reach`).

Why the split: the old design mixed the CCHE1D build (junction detection, node
typing, computing-order renumbering) into `Watershed`, so the generic topology
and the model-specific encoding could not be tested or reused separately. The
split makes `network.py` a standalone, model-accurate encoder that can be fed
from any source, and `Watershed` a thin, testable import + preparation layer.

## Module map

| Module | Role |
|---|---|
| `__init__.py` | Package exports: `Node`, `CrossSection`, `Reach`, `Cell`, `Watershed`, and the three `plot_*` functions. |
| `node.py` | `Node`: a point in a reach. Coordinates (`x/y`, `row/col`), connectivity (`usid`/`dsid`/`us2id`), `type`, `csid`, and the generic inflow source sets (`inflow_cell_sources`, `inflow_reach_sources`). |
| `reach.py` | `Reach`: a chain of `Node`s (a polyline), `id`, `receiving_reach_id`, `slope`, `strahler_number`, `ignore` flag, `us_nd_id`/`ds_nd_id`; helpers for resampling, length, US/DS node order, and plotting. |
| `cell.py` | `Cell`: a TopAGNPS contributing sub-basin cell (`id`, `area`, `receiving_reach_id`, `type` derived from the id's last digit: `…1`=source, `…2`=right, `…3`=left). |
| `crosssection.py` | `CrossSection`: a transverse plane profile — `ws`/`zs` stations/elevations (left→right), the four 1-based feature indices `lfp_idx`/`lbt_idx`/`rbt_idx`/`rfp_idx`, Manning's n (scalar or per-point), trapezoid-with-floodplain generator, thalweg get/set, and plotting. |
| `tools.py` | Low-level, no-`Watershed` helpers: ESRI-ASCII raster reader (`read_esri_asc_file`), geotransform math (`rowcol2latlon_esri_asc`), path-graph skeleton extraction, spline/node resampling (`interpolate_xy_cord_step_numpoints`), the **`custom_dfs_traversal_sorted_predecessors`** DFS that defines CCHE1D computing order, the TopAGNPS CSV section readers, and the legacy `convert_topagnps_output_to_cche1d_input` / `write_cche1d_dat_file` (the pre-object-orientation pipeline, kept for the reference data in `data_outputs/`). |
| `watershed.py` | `Watershed`: the orchestrator — see the two-layer description above. |
| `network.py` | `Cche1dNetwork`, builder, validator, reader, writers — see
  [03_network_builder](03_network_builder.md). **Stdlib-only.** |
| `plotting.py` | `plot_cche1d_network`, `plot_watershed`, `plot_cross_section` (holoviews). |

## Data flow

### Manual path (no TopAGNPS)

```
Reach/Node objects (hand-built or from any source)
    |
    v
build_network_from_reaches(reaches, strahler_drop=0, resample_step=...,
                           cs_template=..., cs_by_generic_id=...)   [network.py]
    |   = _kept_ids + head-node classification + ChannelNetworkBuilder.build()
    v
Cche1dNetwork  (nodes/reaches/channels/links/csecs + channel_heads +
                node_origin/node_provenance + removed_reaches)
    |
    +--> validate_network(net)                      (errors raise)
    +--> write_cche1d_dat_files(net, dir, prefix)   (6 core files)
    +--> write_inflow_reaches_csv(net, path)        (removed-reach record)
```

### TopAGNPS path

```
import_topagnps_reaches_network(csv)   # Reach objects + connectivity
import_topagnps_cells(csv)             # Cell objects (type from id suffix)
read_reaches_geometry_from_topagnps_asc_file(asc)
    # 1. mask NODATA  2. per reach: path graph -> extremities -> node sequence
    # -> generic nodes (US/DS ids)  3. XY from the geotransform
    # 4. update_graph  5. determine_reaches_us_ds_direction
assign_strahler_number_to_reaches()
ignore_reaches_with_strahler_leq(t)    # coarsening (sets reach.ignore)
update_watershed()                     # rebuild graph + expand single-node
                                       # interior reaches (2 midpoint nodes)
resample_reaches(id_list, step=...)    # optional: re-space nodes
assign_cross_section_to_all_points()   # optional: per-node cross sections
build_cche1d_network(strahler_drop=0, resample_step=...)
    # = update_watershed + identify_inflow_sources + build_network_from_reaches
write_cche1d_dat_files(casename, output_folder)
    # 6 core .dat (network.py writer) + *_tw_cell.dat + *_inflow_reaches.csv
```

## Invariants the build relies on

1. Every kept reach has a complete US→DS `dsid` chain and `us_nd_id`/`ds_nd_id`
   that exist in its own `nodes`.
2. A single-node reach with **both** neighbours kept is expanded by
   `update_watershed` (two midpoint nodes) so the builder always sees a ≥2-node
   chain. A single-node reach at a **boundary** stays a **1-point channel** —
   legal (the link carries the `-1` reach-range sentinel).
3. The kept sub-network is connected with exactly one outlet. This holds because
   Strahler order is monotone downstream and coarsening only removes whole
   upstream subtrees.
4. Junctions have at most 5 inflow branches (`MAX_JUNCTION_BRANCHES`), the
   CCHE1D limit.
5. The removed-reach inflow record and the head-node typing share one keep rule
   (`network._kept_ids`), so they can never disagree (see
   [04_modifying_the_network](04_modifying_the_network.md)).
