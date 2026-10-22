# CCHE1D documentation

All documentation for the CCHE1D work lives in this folder (branch `cche1d-consolidation`). It covers the `topagnps2cche1d` library:
everything that turns a channel network — built **by hand** or **imported from
TopAGNPS/AnnAGNPS** — into CCHE1D input the Fortran model and the GUI accept.

## Read in this order

| # | File | Contents |
|---|------|----------|
| 1 | [01_cche1d_input_format](01_cche1d_input_format.md) | **What CCHE1D expects.** The six `.dat` files, node-type codes, the junction block, cross sections, computing order, and what the solver actually uses — distilled from the Fortran readers and the known-good files. |
| 2 | [02_architecture](02_architecture.md) | **What each module does and its scope.** The two-layer split (`Watershed` vs `network.py`), the frozen `Reach`/`Node` contract, a per-module map, and the data flow for both paths. |
| 3 | [03_network_builder](03_network_builder.md) | **The builder API.** `Cche1dNetwork`, `build_network_from_reaches` / `ChannelNetworkBuilder`, `validate_network`, the reader, the writers, and the removed-reach inflow record. |
| 4 | [04_modifying_the_network](04_modifying_the_network.md) | **Manipulating a network.** Strahler coarsening, head-node classification, the removed-reach (inflow) record (manual + TopAGNPS), resampling, cross sections, hydraulic structures. |
| 5 | [05_quickstart](05_quickstart.md) | **Getting started.** Copy-paste examples for the manual path and the TopAGNPS path, plotting, and a verification checklist. |

## The library in one paragraph

`topagnps2cche1d` converts a channel network into the six CCHE1D `.dat` input
files (nodes / reach / channel / link / csec / csprf) plus two auxiliary files
(`*_tw_cell.dat`, `*_inflow_reaches.csv`). The network is built from **generic
`Reach`/`Node` objects** — either hand-written or produced by the TopAGNPS
importer — and encoded by `network.py`, which is the single source of truth for
CCHE1D node typing, the junction block, the computing order, and validation. A
hand-built network and an imported one are indistinguishable to the Fortran
reader. Coarsening (Strahler drop or `ignore` flag) keeps the removed-reach
inflow record in every case, so the model never silently loses upstream runoff.

## Scope and ground truth

- **`network.py` owns the CCHE1D architecture** (build / validate / write /
  read) and is stdlib-only. **`watershed.py` owns the TopAGNPS side** (import,
  geometry, orientation, Strahler, coarsening, inflow bookkeeping) and delegates
  the CCHE1D build to `network.py`. **`reach.py`/`node.py` are the frozen shared
  contract.** See [02_architecture](02_architecture.md).
- **The measure of success is that the Fortran model imports what this library
  writes.** The reference implementation is the CCHE1D Fortran reader
  (`~/projects/CCHE/cche1d_intelfortran/Flow/db_w_ntw_in.f90`,
  `db_w_c1d_in.f90`, `db_w_cs_in.f90`) and the GUI header
  (`~/projects/CCHE/cche1d_gui-vs2019/MyHeader.h`). `network.validate_network`
  mirrors the checks that reader performs, split into errors (fatal) and
  warnings (tolerated, e.g. an unmatched `ND_CSID` or an empty `_csec.dat`).
- The Fortran model and GUI sources under `~/projects/CCHE/` are **read-only**;
  only the Python library is modified.

## Verification (run these, in this order)

```bash
cd ~/projects/topagnps2cche1d

# 1. Fast suite (seconds): network units + ground-truth fixtures +
#    generic-reach topology/bookkeeping + manual parabolic + manual removed-reach
#    inflow + full Ohio e2e.
pixi run -e test test

# 2. Manual-network example script (no TopAGNPS): builds, coarsens, writes the
#    six .dat + inflow CSV, round-trips.
pixi run -e test python notebooks_and_scripts/quickstart_manual_network.py out/manual

# 3. (Optional, ~18 min raster load) the large Goodwin Creek 1 m dataset.
pixi run -e test test_slow
```

The ground-truth CCHE1D input files used by the tests live in the repo:
`data_inputs/netchanTest_example_cche1d_input_files/` (complete) and
`documents/LoopTest/` (nodes with an empty `_csec.dat` — a degenerate case the
model tolerates, used to keep the validator from over-rejecting).

## Assumptions

- Python 3.10+ with the `pixi test` environment (GDAL, pandas, networkx,
  holoviews, …). `network.py` itself is stdlib-only and runs on a bare
  `python3`.
- TopAGNPS output files: `AnnAGNPS_Reach_Data_Section.csv`,
  `AnnAGNPS_Cell_Data_Section.csv`, `AnnAGNPS_Reach_IDs.asc`.
