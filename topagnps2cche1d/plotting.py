"""
Interactive plotting for CCHE1D channel networks and the generic watershed.

Every function returns a ``holoviews`` element (or an overlay of them) so it can
be displayed in a Jupyter notebook, exported to PNG/SVG, or embedded in a panel.
No global plotting state is kept.

Two layers:

* :func:`plot_cche1d_network` — the *built* network (``network.Cche1dNetwork``):
  channels as lines, nodes as points coloured by ``ND_TYPE``, junctions marked.
* :func:`plot_watershed` — the *generic* imported topology (``Watershed``):
  kept reaches as lines, coloured by Strahler order.

Cross sections have their own plotter (:func:`plot_cross_section`) because they
live in the transverse plane, not on the map.

These are standalone so the module can be imported without pulling in the
``Watershed`` (which already owns its own ``Watershed.plot`` for the generic
topology); a ``Cche1dNetwork`` is a pure data object and needs no back-reference.
"""

import pandas as pd
import holoviews as hv
import hvplot.pandas  # noqa: F401  (adds .hvplot to DataFrame)

# CCHE1D node type codes -> short label (for the legend / hover).
NODE_TYPE_LABELS = {
    0: "Source",
    1: "Inflow (BC)",
    2: "Junction",
    3: "Tail (junction)",
    6: "Connection",
    9: "Outlet",
    11: "Structure (11)",
    12: "Structure (12)",
    13: "Structure (13)",
    14: "Structure (14)",
    15: "Structure (15)",
}


def _type_label(t):
    try:
        return NODE_TYPE_LABELS.get(int(t), f"Type {int(t)}")
    except Exception:
        return "Type ?"


def plot_cche1d_network(
    net,
    *,
    title="",
    aspect="equal",
    frame_width=1000,
    frame_height=1000,
    line_width=1.5,
    show_junctions=True,
    hover_channel=("CH_ID",),
):
    """
    Plot a built ``Cche1dNetwork``.

    Channels are drawn as lines (one per ``ChannelSpec``) and nodes as points
    coloured by ``ND_TYPE``.  Junction / tail nodes (type 2 / 3) can be
    overplotted larger for emphasis.  Returns a ``holoviews`` overlay.
    """
    channels = getattr(net, "channels", [])
    nodes = getattr(net, "nodes", [])
    reaches = getattr(net, "reaches", [])
    by_id = {n.nd_id: n for n in nodes}

    # Every CCHE1D reach is one node->node segment; collecting all endpoints
    # per segment and plotting by segment reproduces the full channel geometry.
    recs = []
    for i, rc in enumerate(reaches):
        a = by_id[rc.ndusid]
        b = by_id[rc.nddsid]
        recs.append({"seg": i, "X": a.xc, "Y": a.yc})
        recs.append({"seg": i, "X": b.xc, "Y": b.yc})
    seg_df = pd.DataFrame(recs)
    channel_plot = seg_df.hvplot.line(
        x="X", y="Y", by="seg", line_width=line_width,
        legend=False, title="Channels",
    )

    nrow = [
        {"X": n.xc, "Y": n.yc, "ND_ID": n.nd_id,
         "TYPE": _type_label(n.nd_type)}
        for n in nodes
    ]
    nod = pd.DataFrame(nrow)
    node_plot = nod.hvplot(
        x="X", y="Y", kind="scatter", color="TYPE", size=9, alpha=0.95,
        hover_cols=["ND_ID", "TYPE"], legend=False,
    )

    overlay = channel_plot * node_plot

    if show_junctions:
        jrow = [
            {"X": n.xc, "Y": n.yc, "ND_ID": n.nd_id,
             "TYPE": _type_label(n.nd_type)}
            for n in nodes if int(n.nd_type) in (2, 3)
        ]
        if jrow:
            junc = pd.DataFrame(jrow)
            overlay = overlay * junc.hvplot(
                x="X", y="Y", kind="scatter", color="TYPE", size=22,
                alpha=0.35, hover_cols=["ND_ID", "TYPE"], legend=False,
            )

    return overlay.opts(
        frame_width=frame_width, frame_height=frame_height,
        aspect=aspect, title=title or "CCHE1D network",
    )


def plot_watershed(w, *, title="", aspect="equal", frame_width=1000,
                   frame_height=1000, line_width=1.0, by_strahler=True):
    """
    Plot the generic watershed topology (the kept reaches) with holoviews.

    Returns the same overlay shape as the previous ``Watershed.plot`` so the
    public behaviour is preserved, but now built from the kept reaches only.
    """
    current_graph = w.current_graph
    plots = []
    for reach_id, reach in w.reaches.items():
        if current_graph is not None and reach_id not in current_graph:
            continue
        plots.append(reach.get_nodes_as_df())

    if not plots:
        return hv.Scatter([]).opts(title=title or "Watershed (empty)")

    dfs = pd.concat(plots, ignore_index=True).sort_values(by="ID")

    by = "Strahler_Number" if by_strahler else "Reach_ID"
    hover = ["TYPE", "US2ID", "USID", "ID", "DSID", "Reach_ID", "Strahler_Number"]

    overlay = (
        dfs.hvplot.line(x="X", y="Y", by=by, hover_cols=hover,
                        line_width=line_width)
        * dfs.hvplot(x="X", y="Y", by=by, kind="scatter", alpha=0.5,
                     hover_cols=hover)
    ).opts(
        frame_width=frame_width, frame_height=frame_height,
        aspect=aspect, title=title or "Watershed",
    )
    return overlay


def plot_cross_section(cs, *, title="", aspect=None, frame_width=800,
                       frame_height=500, line_width=2, marker_size=7):
    """
    Plot a single ``CrossSection`` in the transverse plane (ws vs zs).
    """
    ws = cs.ws
    zs = cs.zs
    overlay = (
        hv.Curve((ws, zs)).opts(line_width=line_width)
        * hv.Scatter((ws, zs)).opts(marker="o", size=marker_size)
    )
    label = title or (f"Cross section {cs.id}" if hasattr(cs, "id") else "Cross section")
    return overlay.opts(
        frame_width=frame_width, frame_height=frame_height,
        aspect=aspect, title=label,
    )
