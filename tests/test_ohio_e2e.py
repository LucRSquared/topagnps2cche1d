"""
End-to-end test on the Ohio TopAGNPS dataset (28 reaches; loads in seconds,
so it is part of the fast suite).  Replays the full pipeline on real TopAGNPS
output files and checks the bookkeeping that the long Goodwin run exercises:

    import CSVs -> read reach raster -> Strahler -> [optional coarsening]
    -> resample -> cross sections -> build CCHE1D network -> validate
    -> tw_cell / inflow_reaches DataFrames -> write .dat files -> round trip

The bookkeeping checks re-derive the expected records from the *generic*
reach/node state and compare them to the DataFrames, so they catch wrong
node references (the failure mode of a corrupted provenance map) without
hard-coding dataset-specific ids.

Run:  pixi run -e test test
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

DATA = ROOT / "data_inputs" / "topagnps_ohio_files"
pytestmark = pytest.mark.skipif(
    not (DATA / "AnnAGNPS_Reach_IDs.asc").exists(),
    reason="Ohio TopAGNPS dataset not present (large raster not committed)")


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
    """Fresh watershed with the same reaches/cells (the import path mutates
    nodes in place, so re-running it on the same object double-adds nodes)."""
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


def test_ohio_full_network_validates(imported):
    w, net = _coarsen_and_build(imported, drop=0)
    errors, warnings = netmod.validate_network(net)
    assert errors == [], errors
    assert warnings == [], warnings
    # the full network keeps every reach present in the raster
    assert len(w.current_graph) == len(w.reaches)


@pytest.mark.parametrize("drop", [1, 2])
def test_ohio_coarsened_bookkeeping(imported, drop, tmp_path):
    w, net = _coarsen_and_build(imported, drop=drop)
    errors, _ = netmod.validate_network(net)
    assert errors == [], errors

    valid_nd = {n.nd_id for n in net.nodes}
    origin = {gid: nid for nid, gid in net.node_origin.items()
              if gid is not None}

    # --- independent recount of the removed-reach records ---
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

    # --- independent recount of the tw_cell (cell drainage) records ---
    n_cell_rows = 0
    for rid in w.current_graph:
        for node in w.reaches[rid].nodes.values():
            if node.inflow_cell_sources:
                n_cell_rows += len(node.inflow_cell_sources)
                assert node.id in origin, (
                    f"cell-draining node {node.id} has no network node")
    df_tw = w.create_cche1d_twcells_df()
    assert len(df_tw) == n_cell_rows
    assert set(df_tw["TW_NDDSID"]) <= valid_nd
    # total drainage area is preserved by the import (ha -> m^2)
    assert float(df_tw["TW_DRAREA"].sum()) == pytest.approx(
        sum(c.area for c in w.cells.values()
            if c.receiving_reach_id in w.current_graph))

    # --- head-node types are consistent with the coarsening ---
    types = Counter(n.nd_type for n in net.nodes)
    assert 9 in types                     # the outlet
    if drop:
        assert types[0] + types[1] >= 1   # removed inflows -> BC heads
    # written tw_cell file matches its DataFrame
    paths = w.write_cche1d_dat_files(casename=f"ohio_d{drop}",
                                     output_folder=str(tmp_path))
    lines = (tmp_path / f"ohio_d{drop}_tw_cell.dat").read_text().splitlines()
    assert int(lines[0]) == len(df_tw)
    assert lines[1].startswith("TW_ID\tTW_NUMB\tTW_NDDSID")
    back = netmod.read_cche1d_network(tmp_path)
    errors2, _ = netmod.validate_network(back)
    assert errors2 == [], errors2


def test_ohio_tw_cell_file_format(imported, tmp_path):
    w, _net = _coarsen_and_build(imported, drop=1)
    w.write_cche1d_dat_files(casename="ohio_fmt",
                             output_folder=str(tmp_path))
    lines = (tmp_path / "ohio_fmt_tw_cell.dat").read_text().splitlines()
    n = int(lines[0])
    assert n == len(lines) - 2
    for line in lines[2:]:
        parts = line.split("\t")
        assert len(parts) == 5
        assert int(parts[1]) in w.cells          # TW_NUMB = cell id
        assert int(parts[2]) > 0                 # TW_NDDSID = network ND_ID
