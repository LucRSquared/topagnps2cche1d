"""
End-to-end test on the Goodwin Creek 1 m TopAGNPS dataset (910 reaches).

This is the same pipeline and the same bookkeeping checks as
``test_ohio_e2e.py``, run on the large real dataset that the Ohio test
represents.  The raster load alone takes ~18 minutes, so this file is NOT part
of the fast ``pixi run -e test test`` suite.  Run it explicitly:

    pixi run -e test test_slow

It auto-skips when the reach-ID raster is not present on disk.
"""

import importlib.util
import sys
from collections import Counter
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
except ImportError:  # pragma: no cover - bare interpreter
    pytest.skip("topagnps2cche1d package deps not available",
                allow_module_level=True)

DATA = ROOT / "data_inputs" / "goodwin_creek_1m"
pytestmark = pytest.mark.skipif(
    not (DATA / "AnnAGNPS_Reach_IDs.asc").exists(),
    reason="Goodwin Creek dataset not present (313 MB raster not committed)")


@pytest.fixture(scope="module")
def imported():
    w = Watershed()
    w.import_topagnps_reaches_network(
        DATA / "AnnAGNPS_Reach_Data_Section.csv")
    w.import_topagnps_cells(DATA / "AnnAGNPS_Cell_Data_Section.csv")
    w.read_reaches_geometry_from_topagnps_asc_file(
        DATA / "AnnAGNPS_Reach_IDs.asc")
    w.assign_strahler_number_to_reaches()
    return w


def _clone(w0):
    import copy
    w = Watershed()
    w.reaches = {rid: copy.deepcopy(r) for rid, r in w0.reaches.items()}
    w.cells = copy.deepcopy(w0.cells)
    w.geomatrix = copy.deepcopy(w0.geomatrix)
    w.update_graph()
    w.assign_strahler_number_to_reaches()
    return w


def _coarsen_and_build(w0, drop):
    w = _clone(w0)
    if drop:
        w.ignore_reaches_with_strahler_leq(drop)
    w.update_watershed()
    w.resample_reaches(id_list="all", step=10)
    w.assign_cross_section_to_all_points()
    w.adjust_cross_sections_elevation_with_slope()
    net = w.build_cche1d_network(strahler_drop=0)
    return w, net


def test_goodwin_full_network_validates(imported):
    w, net = _coarsen_and_build(imported, drop=0)
    errors, warnings = netmod.validate_network(net)
    assert errors == [], errors
    assert warnings == [], warnings
    assert len(w.current_graph) == len(w.reaches)
    # one outlet, every node carries a cross section
    types = Counter(n.nd_type for n in net.nodes)
    assert types[9] == 1
    assert all(n.csid > 0 for n in net.nodes)


def test_goodwin_coarsened_bookkeeping(imported, tmp_path):
    # the coarsening level used by the Goodwin notebook
    drop = imported.get_highest_strahler_number() - 2
    w, net = _coarsen_and_build(imported, drop=drop)
    errors, _ = netmod.validate_network(net)
    assert errors == [], errors

    valid_nd = {n.nd_id for n in net.nodes}
    origin = {gid: nid for nid, gid in net.node_origin.items()
              if gid is not None}

    # independent recount of the removed-reach records
    expected_in = []
    for rid in w.current_graph:
        for node in w.reaches[rid].nodes.values():
            for s in node.inflow_reach_sources:
                expected_in.append((s, origin.get(node.id, node.id)))
    df_in = w.get_list_of_inflow_reaches_df()
    got_in = [(int(a), int(b)) for a, b in
              df_in[["inflow_topagnps_reach_id", "receiving_node_id"]]
              .itertuples(index=False)] if not df_in.empty else []
    assert sorted(expected_in) == sorted(got_in)
    assert set(b for _, b in got_in) <= valid_nd
    assert len(df_in) > 0            # coarsening must produce BC heads

    # independent recount of the tw_cell (cell drainage) records
    n_cell_rows = 0
    for rid in w.current_graph:
        for node in w.reaches[rid].nodes.values():
            if node.inflow_cell_sources:
                n_cell_rows += len(node.inflow_cell_sources)
                assert node.id in origin
    df_tw = w.create_cche1d_twcells_df()
    assert len(df_tw) == n_cell_rows
    assert set(df_tw["TW_NDDSID"]) <= valid_nd

    # write + round trip
    paths = w.write_cche1d_dat_files(casename=f"goodwin_d{drop}",
                                     output_folder=str(tmp_path))
    lines = (tmp_path / f"goodwin_d{drop}_tw_cell.dat").read_text().splitlines()
    assert int(lines[0]) == len(df_tw)
    back = netmod.read_cche1d_network(tmp_path)
    errors2, _ = netmod.validate_network(back)
    assert errors2 == [], errors2
