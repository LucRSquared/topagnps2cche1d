# Quickstart

Two ways in: build a network **by hand** (manual) or **import TopAGNPS** output.
Both end at the same place — a `Cche1dNetwork` that `validate_network` accepts
and `write_cche1d_dat_files` turns into the six `.dat` files the Fortran model
imports. See [01_cche1d_input_format](01_cche1d_input_format.md) for what those
files are, [02_architecture](02_architecture.md) for the module map, and
[03_network_builder](03_network_builder.md) for the full builder API.

## Manual path (no TopAGNPS)

Build a small junction of four reaches, give it cross sections, coarsen it, and
write everything. A runnable version lives at
`topagnps2cche1d/notebooks_and_scripts/quickstart_manual_network.py`.

```python
from topagnps2cche1d.node import Node
from topagnps2cche1d.reach import Reach
from topagnps2cche1d.network import (
    build_network_from_reaches, validate_network,
    write_cche1d_dat_files, write_inflow_reaches_csv,
    inflow_reaches_records,
)

def make_reach(rid, recv, pts):
    """A generic reach: a US->DS chain of nodes discharging into `recv`."""
    r = Reach(id=rid, receiving_reach_id=recv, slope=0.0008)
    nodes = [Node(id=100 * rid + k, x=x, y=y) for k, (x, y) in enumerate(pts)]
    for i, n in enumerate(nodes):
        n.dsid = nodes[i + 1].id if i + 1 < len(nodes) else None
        n.usid = nodes[i - 1].id if i > 0 else None
        r.add_node(n)
    r.us_nd_id, r.ds_nd_id = nodes[0].id, nodes[-1].id
    return r

J = (30.0, 60.0); D = (50.0, 100.0)
reaches = [
    make_reach(1, 2,   [(0.0, 0.0), (15.0, 20.0), J]),     # A
    make_reach(3, 2,   [(30.0, 0.0), J]),                  # C
    make_reach(2, 4,   [J, (40.0, 80.0), D]),              # B
    make_reach(4, None,[D, (60.0, 140.0), (70.0, 160.0)]), # outlet
]

# 1. Build the full network (no coarsening).  A manual reach with no Strahler
#    order is kept, so all four reaches are built and it validates.
net = build_network_from_reaches(reaches)
errors, _ = validate_network(net)
assert not errors

# 2. Coarsen: drop two tributaries by flagging them.  Their water now enters at
#    B's head, which becomes a type-0 source node, and the inflow record lists
#    both removed reaches feeding B's head.
reaches[0].ignore = True   # A
reaches[1].ignore = True   # C
net = build_network_from_reaches(reaches)
errors, _ = validate_network(net)
assert not errors
print(inflow_reaches_records(net))   # [(1, <B head ND_ID>), (3, <B head ND_ID>)]

# 3. Write the six core files + the removed-reach inflow CSV.
write_cche1d_dat_files(net, "out", prefix="mynetwork")
write_inflow_reaches_csv(net, "out/mynetwork_inflow_reaches.csv")
```

That is the whole manual workflow. `strahler_drop=t` coarsens by Strahler order
instead of the `ignore` flag (use it when your reaches carry
`strahler_number`, e.g. after `Watershed.assign_strahler_number_to_reaches`).
`cs_template` (a 4-tuple or a `CrossSection`) and `cs_by_generic_id` add cross
sections (see [04_modifying_the_network](04_modifying_the_network.md)).

## TopAGNPS path

Minimum input: three files from one TopAGNPS/AnnAGNPS delineation.

| File | Role |
|---|---|
| `AnnAGNPS_Reach_Data_Section.csv` | one row per reach: `Reach_ID`, `Receiving_Reach`, slope, Manning n, … |
| `AnnAGNPS_Cell_Data_Section.csv` | one row per contributing cell: `Cell_ID`, `Receiving_Reach`, area |
| `AnnAGNPS_Reach_IDs.asc` | ESRI raster: each reach's pixels labelled by its reach id |

```python
import topagnps2cche1d as t2c
from topagnps2cche1d.network import read_cche1d_network, validate_network

root = "data_inputs/topagnps_ohio_files"
w = t2c.Watershed()

w.import_topagnps_reaches_network(f"{root}/AnnAGNPS_Reach_Data_Section.csv")
w.import_topagnps_cells(f"{root}/AnnAGNPS_Cell_Data_Section.csv")
# the slow step: one path-graph pass per reach (~18 min for 910 reaches at 1 m)
w.read_reaches_geometry_from_topagnps_asc_file(f"{root}/AnnAGNPS_Reach_IDs.asc")

w.assign_strahler_number_to_reaches()
w.ignore_reaches_with_strahler_leq(1)            # coarsen (optional)
w.update_watershed()                             # rebuild graph + expand nodes

w.resample_reaches(id_list="all", step=10)       # optional
w.assign_cross_section_to_all_points()           # optional
w.adjust_cross_sections_elevation_with_slope()   # optional

net = w.build_cche1d_network(strahler_drop=0)    # builds + validates
paths = w.write_cche1d_dat_files(casename="ohio", output_folder="out")
# 6 core .dat + out/ohio_tw_cell.dat + out/ohio_inflow_reaches.csv

back = read_cche1d_network("out")
errors, _ = validate_network(back)
assert not errors
```

## Plotting

```python
from topagnps2cche1d import plot_cche1d_network, plot_watershed, plot_cross_section
plot_watershed(w)        # generic topology, coloured by Strahler order
plot_cche1d_network(net) # built network: channels + typed nodes
plot_cross_section(cs)   # one CrossSection, transverse plane
w.plot()                 # == plot_watershed(w)
```

All plot functions return `holoviews` elements (display in Jupyter, export to
PNG/SVG, embed in a panel).

## Verification checklist

After any build, confirm:

1. `validate_network(net)` returns no errors (it raises on errors).
2. Node types are sane: exactly one outlet (9); junctions (2) only where ≥2
   channels converge; inflow boundaries (1) and sources (0) only where upstream
   reaches were removed (see [04](04_modifying_the_network.md)).
3. The removed-reach record is what you expect — `w.get_list_of_inflow_reaches_df()`
   or `network.inflow_reaches_records(net)` — and matches the nodes you
   coarsened.
4. The six core files re-read with `read_cche1d_network` and re-validate.
5. Counts add up: reach rows == sum of per-channel interior segments; node count
   == sum of per-channel point counts.
