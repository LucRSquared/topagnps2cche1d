"""
Exercise the Reach/Node -> ChannelNetworkBuilder adapter with the REAL
topagnps2cche1d classes (imports numpy/pandas/holoviews; runs only inside
the pixi environment — auto-skipped on a bare interpreter).
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topagnps2cche1d_network", ROOT / "topagnps2cche1d" / "network.py")
net = importlib.util.module_from_spec(SPEC)
sys.modules["topagnps2cche1d_network"] = net
SPEC.loader.exec_module(net)

try:
    from topagnps2cche1d.node import Node
    from topagnps2cche1d.reach import Reach
except ImportError:  # pragma: no cover - bare interpreter
    import pytest
    pytest.skip("topagnps2cche1d package deps not available", allow_module_level=True)


def _reach(rid, receiving, pts):
    r = Reach(rid, receiving_reach_id=receiving)
    nid = 100 * rid
    prev = None
    for x, y in pts:
        n = Node(id=nid, x=x, y=y)
        if prev is not None:
            prev.dsid = n.id
        r.add_node(n)
        prev = n
        nid += 1
    r.us_nd_id = 100 * rid
    r.ds_nd_id = prev.id
    return r


SPEC2 = {
    1: (None, [(0, 0), (5, 0), (10, 0)]),
    2: (1, [(20, -20), (15, -10), (10, 0)]),
    3: (1, [(20, 20), (15, 10), (10, 0)]),
}


def test_real_reach_objects_match_dict_api():
    dict_net = net.ChannelNetworkBuilder(SPEC2, resample_step=4).build()
    reach_net = net.ChannelNetworkBuilder(
        [_reach(rid, spec[0], spec[1]) for rid, spec in SPEC2.items()],
        resample_step=4).build()
    assert net.validate_network(reach_net) == []
    a = [(x.nd_id, x.frmno, x.nd_type, x.dsid, x.usid, x.us2id, x.xc, x.yc)
         for x in dict_net.nodes]
    b = [(x.nd_id, x.frmno, x.nd_type, x.dsid, x.usid, x.us2id, x.xc, x.yc)
         for x in reach_net.nodes]
    assert a == b
