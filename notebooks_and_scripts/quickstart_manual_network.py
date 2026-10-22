"""
Manual CCHE1D network — no TopAGNPS inputs.

Build a small hand-written channel network from the generic ``Reach``/``Node``
objects, give every node a cross section, coarsen it (drop the two order-1
tributaries), and write the six core ``.dat`` files *plus* the removed-reach
``inflow_reaches`` CSV.  This is the manual counterpart of the TopAGNPS
import pipeline: the same ``network.py`` builder, writer and validator run
either way, so the output is indistinguishable to the Fortran model.

Run (pixi environment):

    cd ~/projects/topagnps2cche1d
    pixi run -e test python notebooks_and_scripts/quickstart_manual_network.py out/manual

Arguments: an optional output directory (default ``out/manual``).
"""

import sys
from pathlib import Path

# Self-locating: put the repo root (the package's parent dir) on sys.path so
# this script runs directly from any working directory.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from topagnps2cche1d.node import Node
from topagnps2cche1d.reach import Reach
from topagnps2cche1d.watershed import Watershed
from topagnps2cche1d.network import (
    validate_network,
    read_cche1d_network,
)


def make_reach(rid, recv, pts, slope=0.0008):
    """A generic reach with a complete US->DS node chain."""
    r = Reach(id=rid, receiving_reach_id=recv, slope=slope)
    nodes = [Node(id=100 * rid + k, x=x, y=y) for k, (x, y) in enumerate(pts)]
    for i, n in enumerate(nodes):
        n.dsid = nodes[i + 1].id if i + 1 < len(nodes) else None
        n.usid = nodes[i - 1].id if i > 0 else None
        r.add_node(n)
    r.us_nd_id, r.ds_nd_id = nodes[0].id, nodes[-1].id
    return r


def main(out_dir):
    J = (30.0, 60.0)     # A + C confluence
    D = (50.0, 100.0)    # B + (outlet reach) confluence
    w = Watershed()
    #        A(1)              C(3)
    #          \              /
    #           B(2) ------ D(4)   (outlet channel 4 has receiving=None)
    w.add_reach(make_reach(1, 2, [(0.0, 0.0), (15.0, 20.0), J]))
    w.add_reach(make_reach(3, 2, [(30.0, 0.0), J]))
    w.add_reach(make_reach(2, 4, [J, (40.0, 80.0), D]))
    w.add_reach(make_reach(4, None, [D, (60.0, 140.0), (70.0, 160.0)]))
    w.update_graph()
    w.assign_strahler_number_to_reaches()

    # A user-defined (parabolic) cross section on every kept node.
    w.assign_cross_section_to_all_points(
        cs_type="user_defined",
        ws=[-14.0, -5.0, -1.0, 1.0, 5.0, 14.0],
        zs=[6.0, 2.0, 0.0, 0.0, 2.0, 6.0],
        lfp_idx=1, lbt_idx=2, rbt_idx=5, rfp_idx=6,
        mannings_roughness=0.035,
    )
    w.adjust_cross_sections_elevation_with_slope()

    # Coarsen: drop the two order-1 tributaries (A and C).  B's head then has
    # two removed inflows and no kept inflow -> a type-0 source node, and the
    # inflow record lists A and C feeding B's head.
    w.ignore_reaches_with_strahler_leq(1)
    w.update_watershed()

    net = w.build_cche1d_network(strahler_drop=0)   # builds + validates
    errors, warnings = validate_network(net)
    assert not errors, errors
    print(f"built network: {len(net.nodes)} nodes, {len(net.channels)} "
          f"channels, {len(net.csecs)} cross sections; "
          f"{len(warnings)} warning(s)")

    out = Path(out_dir)
    paths = w.write_cche1d_dat_files(casename="manual",
                                     output_folder=str(out))
    for role, p in paths.items():
        print(f"  {role:12s} {p}")

    # A pure manual build (no Watershed) would emit the same CSV directly
    # from the built network, with the shared network-layer writer:
    #   write_inflow_reaches_csv(net, out / "manual_inflow_reaches.csv")
    df = w.get_list_of_inflow_reaches_df()
    print(df.to_string(index=False))

    # Round-trip check: re-read the six core files and re-validate.
    back = read_cche1d_network(out)
    errors, warnings = validate_network(back)
    assert not errors, errors
    print("round-trip: OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/manual")
