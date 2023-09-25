from __future__ import annotations
from pathlib import Path
import os
import csv
import pandas as pd
import numpy as np

from math import ceil

# import scipy
from scipy.interpolate import make_interp_spline

import plotly.express as px
import plotly.graph_objects as go
from osgeo import gdal
from affine import Affine

import networkx as nx


def read_first_line(filename):
    with open(filename, "r") as f:
        r = csv.reader(f)
        num = next(r)
        num = int(num[0])
    return num


def read_cche1d_dat_file(filename):
    numlines = read_first_line(filename)
    df = pd.read_csv(filename, skiprows=1, nrows=numlines, sep="\t", header="infer")
    df.reset_index(drop=True, inplace=True)
    return df


def read_cche1d_channel_dat(filename):
    return read_cche1d_dat_file(filename)


def read_cche1d_csprf_dat(filename):
    return read_cche1d_dat_file(filename)


def read_cche1d_reach_dat(filename):
    return read_cche1d_dat_file(filename)


def read_cche1d_link_dat(filename):
    return read_cche1d_dat_file(filename)


def read_csec_dat(filename):
    return read_cche1d_dat_file(filename)


def cche1d_annagnps_reach_eq(img_reach_old, img_reach_new, nodataval=None):
    # This function returns a dictionary of cche1d reaches and their old AnnAGNPS equivalent
    return {
        new: old
        for new, old in zip(img_reach_new.ravel(), img_reach_old.ravel())
        if new != nodataval
    }


def dict_cche1d_annagnps_to_df(dico):
    return pd.DataFrame(dico.items(), columns=["CCHE1D_Channel", "TopAGNPS_Reach"])


# the nodes file is slightly different
def read_cche1d_nodes_dat(filename):
    numlines = read_first_line(filename)
    df = pd.read_csv(
        filename, skiprows=1, nrows=numlines, sep="\t", header="infer", index_col=False
    )
    df.reset_index(drop=True, inplace=True)
    try:
        df.drop(labels="Unnamed: 11", axis=1, inplace=True)
    except:
        pass
    return df


def read_reach_data_section(filename):
    """
    This function reads an "AgFlow_Reach_Data.csv" file produced by TopAGNPS
    """
    df = pd.read_csv(filename, header="infer", index_col=False)
    df.reset_index(drop=True, inplace=True)
    return df


def read_agflow_reach_data(filename):
    """
    This function reads an "AgFlow_Reach_Data.csv" file produced by TopAGNPS
    """
    df = pd.read_csv(filename, header="infer", index_col=False)
    df.reset_index(drop=True, inplace=True)
    return df


def read_cell_data_section(filename):
    """
    This function reads an "AnnAGNPS_Cell_Data_Section.csv" file produced by TopAGNPS
    """
    return pd.read_csv(filename, index_col=False)


def augment_df_agflow_with_XY(df_agflow, geoMatrix):
    df_agflow[["y_US", "x_US"]] = df_agflow.apply(
        lambda row: pd.Series(
            rowcol2latlon_esri_asc(
                geoMatrix,
                row["Upstream_End_Row"],
                row["Upstream_End_Column"],
                oneindexed=True,
            )
        ),
        axis=1,
    )
    df_agflow[["y_DS", "x_DS"]] = df_agflow.apply(
        lambda row: pd.Series(
            rowcol2latlon_esri_asc(
                geoMatrix,
                row["Downstream_End_Row"],
                row["Downstream_End_Column"],
                oneindexed=True,
            )
        ),
        axis=1,
    )
    return df_agflow


def assemble_agflow_reach_cche1d(df_agflow, df_top_cche1d, geoMatrix, network=None):
    """
    - df_top_cche1d contains two columns 'CCHE1D_Channel' and 'TopAGNPS_Reach'
    - df_agflow is the final dataframe representing the original df_agflow file
    """

    if network is None:
        network = build_network(df_agflow)

    combined = df_top_cche1d.copy(deep=True)

    combined = combined.merge(df_agflow, left_on="TopAGNPS_Reach", right_on="Reach_ID")

    combined[["Upstream_Reach_ID_1", "Upstream_Reach_ID_2"]] = combined.apply(
        lambda x: pd.Series((network[x["Reach_ID"]][0], network[x["Reach_ID"]][1]))
        if x["Reach_ID"] in network.keys()
        else pd.Series((None, None)),
        axis=1,
    )

    # combined = combined.merge(df_agflow[['Reach_ID', 'Receiving_Reach']], left_on='Reach_ID', right_on='Receiving_Reach')
    # combined.rename(columns={'Reach_ID_x': 'Reach_ID', 'Reach_ID_y': 'Upstream_Reach_ID'}, inplace=True)
    # combined.drop(columns=['Receiving_Reach_x', 'Receiving_Reach_y', 'Receiving_Reach'], inplace=True)

    combined = augment_df_agflow_with_XY(combined, geoMatrix)
    combined.drop(
        columns=[
            "TopAGNPS_Reach",
            "Upstream_End_Row",
            "Downstream_End_Row",
            "Upstream_End_Column",
            "Downstream_End_Column",
            "Receiving_Reach",
            "Average_Elevation_[m]",
            "Reach_Length_[m]",
            "Reach_Slope_[m/m]",
            "Upstream_End_UTMx",
            "Upstream_End_UTMy",
            "Downstream_End_UTMx",
            "Downstream_End_UTMy",
            "Distance_Upstream_End_to_Outlet_[m]",
            "Distance_Downstream_End_to_Outlet_[m]",
        ],
        inplace=True,
    )

    return combined


def path_crawler(rowcols):
    """
    This function takes as input a list of (row, col) coordinates
    forming a linear path and returns the linear path connecting them
    prioritizing North/South/East/West direction over diagonals
    """
    deltas = {
        (-1, 0): "North",
        (0, -1): "West",
        (1, 0): "South",
        (0, 1): "East",
        (-1, -1): "NorthWest",
        (1, -1): "SouthWest",
        (1, 1): "SouthEast",
        (-1, 1): "NorthEast",
    }

    G = nx.Graph()

    rowcols_to_visit = rowcols.copy()
    maxiter = len(rowcols_to_visit)
    orphans = [] # The algorithm will most likely not start by an actual extremity
    # Therefore the path will start at some pixel in between, the algorithm will crawl
    # to one end and then will continue from where it started and finish connecting the path
    # in the other direction

    rowcol = rowcols_to_visit.pop(0)
    start = rowcol

    if not rowcols_to_visit:
        # only one node
        G.add_node(rowcol)

    iter = 0
    while rowcols_to_visit:
        iter += 1
        if iter == maxiter:
            raise Exception(
                "Maximum number of iterations reached! Probably invalid path"
            )

        for counter, delta in enumerate(deltas, 1):
            rowcolnext = (rowcol[0] + delta[0], rowcol[1] + delta[1])

            if rowcolnext in rowcols_to_visit:
                G.add_edge(rowcol, rowcolnext)
                # jump there
                rowcol = rowcolnext
                rowcols_to_visit.remove(rowcolnext)
                break
            elif counter == 8:
                # if all directions have been found and there are still rowcols to visit
                # it means the starting point was not an extremity. Therefore we need to start
                # somewhere else and append the orphans that we'll deal with later
                orphans.append(start)
                rowcol = rowcols_to_visit.pop(0)
                start = rowcol

    # Finish connecting the orphans
    for orphan in orphans:
        for delta in deltas:
            rowcolnext = (orphan[0] + delta[0], orphan[1] + delta[1])
            if rowcolnext in rowcols:
                G.add_edge(orphan, rowcolnext)

    return G

def find_extremities_pathgraph(G):
    """
    G: NetworkX graph where all nodes are connected with a maximum of 2 neighbors per node
       so that the final graph is a line graph such as: o----o----o----o----o

    Returns: List of node(s) with a degree inferior or equal to 1 (0 = mono-node graph)
    """
    extremities = [node for node in G.nodes() if nx.degree(G, node) <= 1]

    return extremities


def get_pathgraph_node_sequence(G, source=None, end=None):
    """
    G: NetworkX graph where all nodes are connected with a maximum of 2 neighbors per node
       so that the final graph is a line graph such as: o----o----o----o----o

    source: Optional, extremity node from which to start the graph traversal.
            If not provided the extremities will be identified and the first one found will be used

    end: Optional, Other end of the pathgraph. If provided, verifies that the last node in the list is indeed that one
    
    Returns:
    List of graph nodes starting at source and in path order
    """

    if source is None:
        for node in G.nodes():
            if nx.degree(G, node) <= 1:
                source = node
                break
    
    if source is None:
        raise Exception("No valid extremities found for this graph")
    elif source not in G.nodes():
        raise Exception(f"The extremity provided ({source}) is not in the graph")
    else:
        pathgraph = nx.dfs_preorder_nodes(G, source)

    if (end is not None) and (end == pathgraph[-1]):
        return pathgraph
    else:
        raise Exception(f"End provided: ({end}) does not match with end found: ({pathgraph[-1]})")

def get_pathgraph_binary(img, output_index_starts_one=False):
    """
    This function takes a binary image as input where pixels form
    a linear path and returns the pathgraph (NetworkX graph) of connected (row,cols) extremities
    - if output_index_start_one = True then the rows and cols of the image
    are assumed to start at (1,1) e.g.
    """
    (rows, cols) = np.nonzero(img)

    rowcols = [(row, col) for row, col in zip(rows, cols)]

    if output_index_starts_one:
        rowcols = [(r + 1, c + 1) for r, c in rowcols]

    G = path_crawler(rowcols)

    return G


def find_extremities_binary(img, output_index_starts_one=False):
    """
    This function takes a binary image as input where pixels form
    a linear path and returns the row/cols extremities
    - if output_index_start_one = True then the rows and cols of the image
    are assumed to start at (1,1) e.g.

    I = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1]])
    OUTPUT : [(0,0), (4,4)]

    """
    (rows, cols) = np.nonzero(img)

    rowcols = [(row, col) for row, col in zip(rows, cols)]

    G = path_crawler(rowcols)

    extremities = find_extremities_pathgraph(G)

    if output_index_starts_one:
        extremities = [(r + 1, c + 1) for r, c in extremities]

    return extremities


def build_network(dfagflow, root=None):
    # For now, nothing is done if more than 2 inflows come at a junction
    # If root is specified then all the downstream reaches are ignored

    network = {}

    # All Reaches that receive a tributary in the network
    receiving_reaches = np.unique(dfagflow["Receiving_Reach"].to_numpy())

    # Build the entire network
    for rr in receiving_reaches:
        candidates = dfagflow.loc[
            (dfagflow["Receiving_Reach"] == rr) & (dfagflow["Reach_ID"] != rr), :
        ]

        candidates = candidates["Reach_ID"].to_list()

        # If reaches A and B are pouring into Reach C then
        # the counterclockwise order of inflows into C (when looking upstream) is A, B
        # and A > B > C

        network[rr] = [max(candidates), min(candidates)]

    if root is not None:
        reaches_to_keep = dfs_iterative_postorder(network, root)

        new_network = {}

        for r in reaches_to_keep:
            if r in network.keys():
                new_network[r] = network[r]

        return new_network

    return network


def reorder_network(network, permutation_vect):
    if len(network) == 0:
        return {}

    reordered_network = {}

    for k in list(network):
        reordered_network[permutation_vect.index(k) + 1] = [
            permutation_vect.index(e) + 1 for e in network.pop(k)
        ]

    return reordered_network


def interpolate_xy_cord_step_numpoints(x, y, **kwargs):
    """
    Given two x and y arrays provides two new arrays xnew, ynew that are interpolated:
    key-value arguments:
        - step : define a step length (in the units of x and y) to resample points along the cord length
        OR
        - numpoints: define an arbitrary number of points to resample along the cord length
    WARNING: If both values are specified, the method yielding the greater of resampling nodes will be chosen
    It is advised to use only one of the keywords arguments.
    """
    if "step" in kwargs:
        step = kwargs["step"]
    else:
        step = None

    if "numpoints" in kwargs:
        numpoints = kwargs["numpoints"]
    else:
        numpoints = 0

    if step is None and numpoints == 0:
        raise Exception("No resampling parameter provided (step, or numpoints)")

    # Arrange coordinates in pairs:
    p = np.stack((x, y))

    dp = p[:, 1:] - p[:, :-1]  # 2 vector distance between points
    l = (dp**2).sum(axis=0)  # squares of lengths of 2-vectors between points
    u_cord = np.sqrt(l).cumsum()  # cumulative sums of 2-norms
    u_cord = np.r_[0, u_cord]  # the first point is parameterized at zero

    # Get arc length
    arc_length = u_cord[-1]  # by definition

    # Compute number of nodes needed for interpolation
    if step is not None:
        numpoints = max(
            ceil(arc_length / step), numpoints
        )  # chooses the highest value between
        # provided numpoints and the one computed
        # with the step method
    # create spline interpolant and do linear if it fails
    try:
        spl_cord = make_interp_spline(u_cord, np.c_[x, y])

        uu = np.linspace(u_cord[0], u_cord[-1], numpoints)
        xx_cord, yy_cord = spl_cord(uu).T
    except ValueError:
        # If spline interpolation fails, fall back to linear interpolation
        xx_cord = np.interp(np.linspace(0, 1, numpoints), np.linspace(0, 1, len(x)), x)
        yy_cord = np.interp(np.linspace(0, 1, numpoints), np.linspace(0, 1, len(y)), y)

    return xx_cord, yy_cord


def get_counterclockwise_inflows(candidates, juncrowcol, img_flovec, oneindexed=False):
    # WARNING The Assumption is made that the orientation of the plane is indirect
    # due to ij orientation j <-> x / i <-> y

    # Candidates is a subselection of the dfagflow dataframe with the list of candidates
    # juncrowcol is a tuple of the upstream (row,col) coordinates of the receiving reach

    cand_reach_ids = [rid for rid in candidates["Reach_ID"]]

    DS_end = [
        x + y * 1j
        for x, y in zip(
            candidates["Downstream_End_Column"], candidates["Downstream_End_Row"]
        )
    ]

    if not oneindexed:
        juncrowcol = np.add(juncrowcol, (1, 1))

    junc_complex = juncrowcol[1] + juncrowcol[0] * 1j

    # Get flovec of outrowcol
    if oneindexed:
        idx_flovec = int(img_flovec[juncrowcol[0] - 1, juncrowcol[1] - 1])
    else:
        idx_flovec = int(img_flovec[juncrowcol[0], juncrowcol[1]])

    refvec_flovec = {
        1: -1 - 1j,
        2: -1j,
        3: 1 - 1j,
        4: -1 + 0j,
        6: 1 + 0j,
        7: -1 - 1j,
        8: +1j,
        9: 1 + 1j,
    }
    # These numbers are so to represent the direction of the flow
    # in a cell because the orientation of the plane is ij not xy
    # See TOPAZ User Manual FLOVEC description

    dsends_minus_junc_row_col = [
        (ds - junc_complex) / refvec_flovec[idx_flovec] for ds in DS_end
    ]

    angles_of_incoming_reaches = np.angle(dsends_minus_junc_row_col, deg=True)
    # Make sure the angles are in the [0, 360 [ range
    angles_of_incoming_reaches = (angles_of_incoming_reaches + 360) % 360

    # Same here, the descending order is taken because of the ij orientation of the plane
    idxsort = np.argsort(angles_of_incoming_reaches)
    idxsort = idxsort[::-1]

    reaches_counterclockwise_order = list(np.array(cand_reach_ids)[idxsort])

    return reaches_counterclockwise_order


def custom_dfs_traversal_sorted_predecessors(
    G, start=None, visit_descending_order=True, postorder=True
):
    """
    Custom DFS traversal of directed graph G given a start node "start". If start node is not provided
    it will use the node with an out degree equal to 0
    If visit_descending_order = True then the predecessors of a given node will be visited in descending order
        (assuming nodes are sortable)
    If postorder is true the nodes will be listed in a postorder fashion
    """
    if start is None:
        # list nodes with out_degree 0 and take the first one
        start = [node[0] for node in G.out_degree() if node[1] == 0][0]

    visited = set()
    result = []

    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            predecessors = sorted(G.predecessors(node), reverse=visit_descending_order)
            unvisited_predecessors = [
                predecessor
                for predecessor in predecessors
                if predecessor not in visited
            ]
            stack.extend(unvisited_predecessors)
            result.append(node)

    if postorder:
        result = result[::-1]

    return result


def dfs_iterative_postorder(network, outlet_reach):
    # Depth First Search Post Order Tree Traversal (Iterative)

    # network is a dict
    # Each entry in the network is a reach and points to the list of inflow reaches ordered from "right to left"

    # approach: use a stack for saving each child
    #           and reverse the result
    # https://lenchen.medium.com/leetcode-590-n-ary-tree-postorder-traversal-3300406214cb

    # Example:
    # network = {0:[1,2], 1:[3,4], 2:[5,6], 6:[7]}
    # start = 0
    # postorder = dfs_iterative_postorder(network, start)

    if len(network) == 0:
        return []

    if outlet_reach not in network:
        return []

    stack = [outlet_reach]
    result = []

    while stack:
        reach = stack.pop()
        result.append(reach)

        if reach not in network:  # if reach has inflows
            inflows = []
        else:
            inflows = network[reach]

        for tributaries in inflows:
            stack.append(tributaries)

    postorder = result[::-1]

    return postorder


def dfs_recursive_postorder(network, reach, postorder=None):
    # Depth First Search Post Order Tree Traversal (Recursive)

    # network is a dict
    # Each entry in the network is a reach and points to the list of inflow reaches ordered from "right to left"
    #
    # Example:
    # network = {0:[1,2], 1:[3,4], 2:[5,6], 6:[7]}
    # start = 0
    # postorder = dfs_recursive_postorder(network, start)

    if postorder == None:
        postorder = []

    if reach not in network:
        postorder.append(reach)
        return postorder

    inflows = network[reach]

    for c in inflows:
        dfs_recursive_postorder(network, c, postorder)

    postorder.append(reach)

    return postorder


def cleanup_merge_min_strahler(
    dfagflow, img_reach_asc, nodataval, img_netw_asc, min_netw
):
    # Removes reaches from dfagflow that have a Strahler number smaller than min_netw
    # Merge residual reaches that no longer have two inflows into a single one (the downstream one)
    # Cleans up img_reach_asc

    dfagflow = dfagflow.copy(deep=True)

    reaches_to_remove_strahler = np.unique(
        np.array(img_reach_asc[img_netw_asc < min_netw])
    )

    reaches_to_remove_strahler = reaches_to_remove_strahler[
        reaches_to_remove_strahler > 0
    ]  # 0 is not a reach

    # Remove from dfagflow
    dfagflow = dfagflow.drop(
        dfagflow[dfagflow["Reach_ID"].isin(reaches_to_remove_strahler)].index
    )
    # Remove from img_reach_asc
    img_reach_asc[img_netw_asc < min_netw] = nodataval

    receiving_reaches = dfagflow["Receiving_Reach"].to_list()
    reaches_with_single_inflow = [
        reach for reach in receiving_reaches if receiving_reaches.count(reach) == 1
    ]

    # for r in reaches_with_single_inflow:

    while reaches_with_single_inflow:
        r = reaches_with_single_inflow[0]  # Get first element
        us_reach_id = dfagflow.loc[dfagflow["Receiving_Reach"] == r, "Reach_ID"].values[
            0
        ]

        # print(f'us reach = {us_reach_id}, ds reach = {r}, number of reaches left: {len(dfagflow)} reaches left to process: {len(reaches_with_single_inflow)}')

        dfagflow.loc[dfagflow["Reach_ID"] == r, "Upstream_End_Row"] = (
            dfagflow.loc[dfagflow["Reach_ID"] == us_reach_id, "Upstream_End_Row"]
            .astype("int")
            .values[0]
        )
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Upstream_End_Column"] = (
            dfagflow.loc[dfagflow["Reach_ID"] == us_reach_id, "Upstream_End_Column"]
            .astype("int")
            .values[0]
        )
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Drainage_Area_[ha]"] = (
            dfagflow.loc[dfagflow["Reach_ID"] == r, "Drainage_Area_[ha]"].values[0]
            + dfagflow.loc[
                dfagflow["Reach_ID"] == us_reach_id, "Drainage_Area_[ha]"
            ].values[0]
        )
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Average_Elevation_[m]"] = 0.5 * (
            dfagflow.loc[dfagflow["Reach_ID"] == r, "Average_Elevation_[m]"].values[0]
            + dfagflow.loc[
                dfagflow["Reach_ID"] == us_reach_id, "Average_Elevation_[m]"
            ].values[0]
        )
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Reach_Length_[m]"] = (
            dfagflow.loc[dfagflow["Reach_ID"] == r, "Reach_Length_[m]"].values[0]
            + dfagflow.loc[
                dfagflow["Reach_ID"] == us_reach_id, "Reach_Length_[m]"
            ].values[0]
        )
        dfagflow.loc[
            dfagflow["Reach_ID"] == r, "Distance_Upstream_End_to_Outlet_[m]"
        ] = dfagflow.loc[
            dfagflow["Reach_ID"] == us_reach_id, "Distance_Upstream_End_to_Outlet_[m]"
        ].values[
            0
        ]
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Reach_Slope_[m/m]"] = np.nan
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Contributing_Cell_ID_Source"] = np.nan
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Contributing_Cell_ID_Left"] = np.nan
        dfagflow.loc[dfagflow["Reach_ID"] == r, "Contributing_Cell_ID_Right"] = np.nan

        dfagflow.loc[dfagflow["Receiving_Reach"] == us_reach_id, "Receiving_Reach"] = r
        img_reach_asc[img_reach_asc == us_reach_id] = r

        # Remove reach r from dfagflow
        dfagflow.drop(dfagflow[dfagflow["Reach_ID"] == us_reach_id].index, inplace=True)

        # Recompute the list of reaches with single inflow
        receiving_reaches = dfagflow["Receiving_Reach"].to_list()
        reaches_with_single_inflow = [
            reach for reach in receiving_reaches if receiving_reaches.count(reach) == 1
        ]

    img_netw_asc = img_netw_asc - min_netw + 1
    img_netw_asc[img_netw_asc < 0] = 0

    return dfagflow, img_reach_asc, img_netw_asc


def convert_topagnps_output_to_cche1d_input(
    filepath_agflow,
    # filepath_flovec,
    filepath_annagnps_reach_ids,
    filepath_netw,
    cross_sections,
    min_netw=1,
    distance=1,
):
    """
    This script takes topagns reaches and changes the numbering according to the CCHE1D numbering
    system for links
    ASSUMPTIONS :
    - Only 2 inflows per junction (TopAGNPS guarantees it but AnnAGNPS is more liberal)
    - The Outlet "reach" is its own reach but has 0 length, potential problem (?)
    Inputs :
    - FILEPATH to Agflow file
    - FILEPATH to AnnAGNPS_Reach_IDs.ASC (DEM file)
    - cross_sections : dictionary containing the cross sections, for now it's just a default
       default_xsection = {'type' : 'default',
                           'CP_Ws': [-43, -35, -13, -10, 10, 13, 35, 43],
                           'CP_Zs': [6, 2, 2, 0, 0, 2, 2, 6]}
    - min_netw : Minimum Strahler Number
    - distance : distance in meters at which interval the reaches are sampled (for cross-sections)
    """

    # Reading
    # img_flovec = read_esri_asc_file(filepath_flovec)[0]
    img_reach_asc, geomatrix, _, _, nodataval_reach_asc, _ = read_esri_asc_file(
        filepath_annagnps_reach_ids
    )
    img_netw_asc, _, _, _, _, _ = read_esri_asc_file(filepath_netw)
    # img_dednm_asc, _, _, _, _, _ = read_esri_asc_file(filepath_dednm)
    dfagflow_original = read_agflow_reach_data(filepath_agflow)

    # The raster is assumed to be a square, thus the raster size (in meters) is:
    rsize = abs(geomatrix[1])

    # FUTURE:
    # Test if cross_sections is non-default and read the cross-sections. I'm not sure in what format yet.
    # Will have to think about when the cross-sections are pre-specified, the rsize/distance sampling of the reaches
    # Will need to be adjusted so that the

    outlet_reach_id = dfagflow_original.loc[
        (dfagflow_original["Distance_Downstream_End_to_Outlet_[m]"] == 0)
        & (dfagflow_original["Reach_Length_[m]"] != 0),
        "Reach_ID",
    ].values[0]

    # Remove Reaches that have a Strahler Number smaller than min_netw
    dfagflow, img_reach_asc, img_netw_asc = cleanup_merge_min_strahler(
        dfagflow_original, img_reach_asc, nodataval_reach_asc, img_netw_asc, min_netw
    )

    # Build the flow network (each node points to the counterclock wise list of tributaries at the upstream junction)
    # network = build_network(dfagflow, img_flovec, root=outlet_reach_id)
    network = build_network(dfagflow, root=outlet_reach_id)

    if network:
        cche1d_reordered_reaches = dfs_iterative_postorder(network, outlet_reach_id)
        reordered_network = reorder_network(network, cche1d_reordered_reaches)
    else:
        reordered_network = {}
        cche1d_reordered_reaches = []

    (
        df_nodes,
        df_channel,
        df_link,
        df_reach,
        df_csec,
        df_csprf,
        img_reach_reordered,
    ) = create_cche1d_tables(
        dfagflow,
        geomatrix,
        img_reach_asc,
        img_netw_asc,
        cche1d_reordered_reaches,
        reordered_network,
        cross_sections,
        rsize,
        distance,
    )

    # Compute cche1d and AnnAGNPS equivalent reaches
    cche1d_to_annagnps_reaches = cche1d_annagnps_reach_eq(
        img_reach_asc, img_reach_reordered, nodataval=nodataval_reach_asc
    )

    # Insert all functions for full package output for CCHE1D
    df_cche1d_to_annagnps_reaches = dict_cche1d_annagnps_to_df(
        cche1d_to_annagnps_reaches
    )

    df_cche1d_annagnps_conn = assemble_agflow_reach_cche1d(
        dfagflow_original, df_cche1d_to_annagnps_reaches, geomatrix
    )

    return (
        df_nodes,
        df_channel,
        df_link,
        df_reach,
        df_csec,
        df_csprf,
        img_reach_reordered,
        df_cche1d_annagnps_conn,
    )


def apply_permutation_int_array(original_arr, permutation_vect):
    renumbered_arr = np.copy(original_arr)

    # Delete entries that aren't in the permutation_vect
    unique_vals = np.unique(renumbered_arr)
    vals_to_replace_by_zero = set(unique_vals) - set(permutation_vect)

    for v in vals_to_replace_by_zero:
        renumbered_arr[renumbered_arr == v] = 0

    for newid, oldid in enumerate(permutation_vect, 1):
        renumbered_arr[original_arr == oldid] = newid

    return renumbered_arr


def apply_permutation_int_dfagflow(dfagflow, permutation_vect):
    # dfagflow_new = dfagflow.copy(deep=True)
    # Keep only the reaches present in the permutation vector

    if permutation_vect:
        dfagflow_new = dfagflow[dfagflow["Reach_ID"].isin(permutation_vect)].copy()

        dfagflow_new["New_Reach_ID"] = dfagflow_new.apply(
            lambda x: permutation_vect.index(int(x["Reach_ID"])) + 1, axis=1
        )
        dfagflow_new["New_Receiving_Reach"] = dfagflow_new.apply(
            lambda x: permutation_vect.index(int(x["Receiving_Reach"])) + 1
            if x["Receiving_Reach"] in permutation_vect
            else -1,
            axis=1,
        )
    else:
        dfagflow_new = dfagflow.copy(deep=True)
        dfagflow_new.rename(
            columns={
                "Reach_ID": "New_Reach_ID",
                "Receiving_Reach": "New_Receiving_Reach",
            },
            inplace=True,
        )

    return dfagflow_new


def create_cche1d_tables(
    dfagflow,
    geomatrix,
    img_reach_asc,
    img_netw_asc,
    permutation_vect,
    reordered_network,
    cross_sections,
    rsize,
    distance,
):
    # INPUTS:
    # - dfagflow: Original AgFlow dataframe
    # - img_reach_asc: Array with pixels numbered according to the old (topagnps) numbering of reaches
    # - img_netw_asc: Array with reaches numbered according to their Strahler order
    # - permuation_vect: permutation vector for new numbering of reaches
    # - cross_sections: dictionary containing cross sections. if cross_sections['type'] == 'default' then all the cross-sections will be identical
    #
    # OUTPUTS:
    # - df_nodes table
    # - df_channel table
    # - df_link table
    # - df_reach table
    # - df_csec table
    # - df_csprf table

    # Apply new numbering to AgFlow Dataframe and img_reach_asc
    dfagflow_new = apply_permutation_int_dfagflow(dfagflow, permutation_vect)

    if permutation_vect:
        img_reach_asc_new = apply_permutation_int_array(img_reach_asc, permutation_vect)
    else:
        img_reach_asc_new = img_reach_asc

    # Initialize Nodes Arrays
    nd_ID = []
    nd_FRMNO = []
    nd_TYPE = []
    nd_XC = []
    nd_YC = []
    nd_DSID = []
    nd_USID = []
    nd_US2ID = []
    nd_CSID = []

    # Initialize Channel Arrays
    ch_ID = []
    ch_NDUSID = []
    ch_NDDSID = []

    # Initialize Link Arrays (!!! Reaches for TopAGNPS are CCHE1D links and channels (in the absence of the hydraulic structure))
    lk_ID = []
    lk_CMPSEQ = []
    lk_NDUSID = []
    lk_NDDSID = []
    lk_RCUSID = []
    lk_RCDSID = []
    # lk_TYPE = []

    # Reach table (!!! For CCHE1D a REACH is not the same thing as a TopAGNPS reach.)
    #                  For CCHE1D a REACH is formed by two points
    rc_ID = []
    rc_NDUSID = []
    rc_NDDSID = []
    rc_ORDER = []
    rc_LENGTH = []

    # CSEC table (default values are being used and defined in the lower portion of the code)
    # cs_ID = []
    # cs_NPTS = []
    # cs_LOB = []
    # cs_LBT = []
    # cs_ROB = []
    # cs_RBT = []
    # cs_SVTYPE = []
    # cs_TYPE = []
    # cs_ORIGIN = []
    # cs_STATION = []

    # CSPRF table
    cp_ID = []
    cp_CSID = []
    cp_POSIDX = []
    cp_W = []
    cp_Z = []
    cp_RGH = []
    # cp_BLOCK = [] # Default values used and defined later

    # Useful indexing tools
    FirstInflowLastNodeIDForReceivingReach = {}
    SecondInflowLastNodeIDForReceivingReach = {}

    FirstInflowLastNodeAbsoluteIndexForReceivingReach = {}
    SecondInflowLastNodeAbsoluteIndexForReceivingReach = {}

    N_max_reach = dfagflow_new["New_Reach_ID"].max()

    if reordered_network:
        first_inflows = [e[0] for e in reordered_network.values()]
        second_inflows = [e[1] for e in reordered_network.values()]
        list_of_receiving_reaches = [k for k in reordered_network.keys()]
    else:
        first_inflows = []
        second_inflows = []
        list_of_receiving_reaches = []

    # outlet_reach = dfagflow_new.loc[dfagflow_new['Reach_Length_[m]']==0,['New_Reach_ID']]
    outlet_reach_id = dfagflow_new.loc[
        (dfagflow_new["Distance_Downstream_End_to_Outlet_[m]"] == 0)
        & (dfagflow_new["Reach_Length_[m]"] != 0),
        "New_Reach_ID",
    ].values[0]

    nd_counter = 1
    xs_nds_counter = 1

    # Goes from 1 to N_max_reach
    for reach_id in range(1, N_max_reach + 1):
        # Get Upstream ROW/COL of current reach
        us_row, us_col = dfagflow_new.loc[
            dfagflow_new["New_Reach_ID"] == reach_id,
            ["Upstream_End_Row", "Upstream_End_Column"],
        ].values[0]

        # Get Downstream ROW/COL of current reach
        ds_row, ds_col = dfagflow_new.loc[
            dfagflow_new["New_Reach_ID"] == reach_id,
            ["Downstream_End_Row", "Downstream_End_Column"],
        ].values[0]

        # Keep only the pixels pertaining to the current reach
        curr_img_reach = np.where(img_reach_asc_new == reach_id, 1, 0)

        # Write function get_intermediate_nodes (-1 is added because the rows/cols in dfagflow are 1-indexed )
        ordered_path = get_intermediate_nodes_img(
            (us_row - 1, us_col - 1),
            (ds_row - 1, ds_col - 1),
            curr_img_reach,
            rsize,
            distance,
        )

        # Computing corresponding coordinates
        YCXC_tmp = [
            rowcol2latlon_esri_asc(
                geomatrix, rowcol[0], rowcol[1], oneindexed=False
            )  # voluntary ondeindexed=False because row/col comes from the previous function that makes it 0 indexed
            for rowcol in ordered_path
        ]
        YC_tmp, XC_tmp = zip(*YCXC_tmp)
        XC_tmp = list(XC_tmp)
        YC_tmp = list(YC_tmp)

        # For all reaches that are not the outlet
        if reach_id != outlet_reach_id:
            # Get receiving reach's upstream node (that will be the junction)
            receiving_reach_id = int(
                dfagflow_new.loc[
                    dfagflow_new["New_Reach_ID"] == reach_id, ["New_Receiving_Reach"]
                ].values[0]
            )
            junc_node_row, junc_node_col = dfagflow_new.loc[
                dfagflow_new["New_Reach_ID"] == receiving_reach_id,
                ["Upstream_End_Row", "Upstream_End_Column"],
            ].values[0]
            junc_node_y, junc_node_x = rowcol2latlon_esri_asc(
                geomatrix, junc_node_row, junc_node_col, oneindexed=True
            )

            # Append to the reach
            XC_tmp.append(junc_node_x)
            YC_tmp.append(junc_node_y)

        numpoints = len(XC_tmp)

        nd_ID_tmp = list(range(nd_counter, nd_counter + numpoints))

        nd_CSID_tmp = nd_ID_tmp.copy()

        # Channel table
        ch_ID.append(reach_id)
        ch_NDUSID.append(nd_ID_tmp[0])
        ch_NDDSID.append(nd_ID_tmp[-1])

        # Link table
        lk_ID.append(reach_id)
        lk_CMPSEQ.append(
            reach_id
        )  # the loop already goes in the computational sequence
        lk_NDUSID.append(nd_ID_tmp[0])
        lk_NDDSID.append(nd_ID_tmp[-1])
        lk_RCUSID.append(reach_id)
        lk_RCDSID.append(reach_id)

        # Reach table
        rc_ID_tmp = list(range(1, numpoints))
        rc_LENGTH_tmp = compute_cche1d_reaches_length(XC_tmp, YC_tmp)

        if reach_id != outlet_reach_id:
            rc_ORDER_tmp = collect_data_along_img_path(
                img_netw_asc, ordered_path, oneindexed=False
            )  # The last point is implicitly ignored since
            # ordered_path is computed before appending the ds junction node
        else:
            rc_ORDER_tmp = collect_data_along_img_path(
                img_netw_asc, ordered_path[0:-1], oneindexed=False
            )  # The outlet node is ignored since it is the downstream node of the last reach

        rc_NDUSID_tmp = nd_ID_tmp[0:-1]
        rc_NDDSID_tmp = nd_ID_tmp[1:]

        # CSPRF table
        if cross_sections["type"] == "default":
            npts = len(cross_sections["CP_Ws"])

            cp_ID_tmp = []
            cp_CSID_tmp = []
            cp_POSIDX_tmp = []
            cp_W_tmp = []
            cp_Z_tmp = []
            cp_RGH_tmp = []
            for n in nd_ID_tmp:
                cp_ID_tmp_tmp = list(range(xs_nds_counter, xs_nds_counter + npts))
                cp_ID_tmp.extend(cp_ID_tmp_tmp)
                cp_CSID_tmp.extend([n for _ in cp_ID_tmp_tmp])
                cp_POSIDX_tmp.extend(list(range(1, npts + 1)))
                cp_W_tmp.extend(cross_sections["CP_Ws"])
                cp_Z_tmp.extend(cross_sections["CP_Zs"])
                cp_RGH_tmp.extend(cross_sections["CP_RGHs"])

                xs_nds_counter += npts  # Increment cross-section nodes counter
        elif cross_sections["type"] == "cutdem":  # DO THE CASE WHERE WE CUT THE DEM
            width = cross_sections["width"]
            # x = np.array([23, 24, 24, 25, 25])
            # y = np.array([13, 12, 13, 12, 13])

            # # append the starting x,y coordinates
            # # x = np.r_[x, x[0]]
            # # y = np.r_[y, y[0]]

            # # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
            # # is needed in order to force the spline fit to pass through all the input points.
            # # tck, u = interpolate.splprep([x, y], per=True, s=0)
            # tck, u = interpolate.splprep([x, y], s=0)

            # # evaluate the spline fits for 1000 evenly spaced distance values
            # xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

        nd_FRMNO_tmp = (
            nd_ID_tmp.copy()
        )  # Those IDs are the same because this procedure is done in the computational order
        nd_US2ID_tmp = [-1 for _ in nd_ID_tmp]
        nd_TYPE_tmp = [
            6 for _ in nd_ID_tmp
        ]  # assign value 6 by default, special cases are treated hereafter

        if reach_id in first_inflows:
            idx1 = first_inflows.index(reach_id)

            FirstInflowLastNodeIDForReceivingReach[
                list_of_receiving_reaches[idx1]
            ] = nd_ID_tmp[-1]
            FirstInflowLastNodeAbsoluteIndexForReceivingReach[
                list_of_receiving_reaches[idx1]
            ] = (len(nd_ID) + len(nd_ID_tmp) - 1)

        if reach_id in second_inflows:
            idx2 = second_inflows.index(reach_id)

            SecondInflowLastNodeIDForReceivingReach[
                list_of_receiving_reaches[idx2]
            ] = nd_ID_tmp[-1]
            SecondInflowLastNodeAbsoluteIndexForReceivingReach[
                list_of_receiving_reaches[idx2]
            ] = (len(nd_ID) + len(nd_ID_tmp) - 1)

            nd_CSID_tmp[-1] = FirstInflowLastNodeIDForReceivingReach[
                list_of_receiving_reaches[idx2]
            ]

        if reach_id == outlet_reach_id:
            nd_TYPE_tmp[-1] = 9  # Label last node as the outflow
            nd_DSID_tmp = nd_ID_tmp[1:]
            nd_DSID_tmp.append(
                nd_DSID_tmp[-1]
            )  # The "downstream" node of the outlet node is itself
        else:
            nd_TYPE_tmp[-1] = 3  # Last node is end of link by default

            nd_DSID_tmp = nd_ID_tmp[1:]
            nd_DSID_tmp.append(None)  # Leave a placeholder (to keep the index correct)
            # until it can be resolved when the algorithm
            # gets to the receiving reach

        if reach_id not in list_of_receiving_reaches:
            # Source reach
            nd_USID_tmp = [-1]  # First node is a source node
            nd_USID_tmp.extend(nd_ID_tmp[0:-1])

            nd_TYPE_tmp[0] = 0  # First node is a source node

        else:
            # Any other reach that receives flow from another one

            # Complete in the main array the DS node id of the correponding inflows
            idx_first_inflow_last_node = (
                FirstInflowLastNodeAbsoluteIndexForReceivingReach[reach_id]
            )
            nd_DSID[idx_first_inflow_last_node] = nd_ID_tmp[
                0
            ]  # The downstream point of the last point of the first inflow is the current receiving node

            idx_second_inflow_last_node = (
                SecondInflowLastNodeAbsoluteIndexForReceivingReach[reach_id]
            )
            nd_DSID[idx_second_inflow_last_node] = nd_ID_tmp[0]

            nd_USID_tmp = [SecondInflowLastNodeIDForReceivingReach[reach_id]]
            nd_USID_tmp.extend(nd_ID_tmp[0:-1])
            nd_US2ID_tmp[0] = FirstInflowLastNodeIDForReceivingReach[reach_id]

            nd_CSID_tmp[0] = FirstInflowLastNodeIDForReceivingReach[reach_id]

            nd_TYPE_tmp[0] = 2  # First node is a junction node

        # Increment nd_counter
        nd_counter += numpoints

        # Append all the temporary lists
        nd_ID.extend(nd_ID_tmp)
        nd_FRMNO.extend(nd_FRMNO_tmp)
        nd_TYPE.extend(nd_TYPE_tmp)
        nd_XC.extend(XC_tmp)
        nd_YC.extend(YC_tmp)
        nd_CSID.extend(nd_CSID_tmp)
        nd_USID.extend(nd_USID_tmp)
        nd_DSID.extend(nd_DSID_tmp)
        nd_US2ID.extend(nd_US2ID_tmp)

        rc_ID.extend(rc_ID_tmp)
        rc_LENGTH.extend(rc_LENGTH_tmp)
        rc_ORDER.extend(rc_ORDER_tmp)
        rc_NDUSID.extend(rc_NDUSID_tmp)
        rc_NDDSID.extend(rc_NDDSID_tmp)

        cp_ID.extend(cp_ID_tmp)
        cp_CSID.extend(cp_CSID_tmp)
        cp_POSIDX.extend(cp_POSIDX_tmp)
        cp_W.extend(cp_W_tmp)
        cp_Z.extend(cp_Z_tmp)
        cp_RGH.extend(cp_RGH_tmp)

    # Default values
    nd_RSID = [-1 for _ in nd_ID]
    nd_STID = [1 for _ in nd_ID]

    lk_TYPE = [1 for _ in lk_ID]

    cp_BLOCK = [1 for _ in cp_ID]

    # cp_BLOCK = []
    if cross_sections["type"] == "default":
        cs_ID = nd_ID.copy()
        cs_NPTS = list(np.full_like(cs_ID, len(cross_sections["CP_Ws"])))
        cs_LOB = list(np.full_like(cs_ID, 1))
        cs_LBT = list(np.full_like(cs_ID, 1))
        cs_ROB = list(np.full_like(cs_ID, len(cross_sections["CP_Ws"])))
        cs_RBT = list(np.full_like(cs_ID, len(cross_sections["CP_Ws"])))
        cs_SVTYPE = ["WZ" for _ in cs_ID]
        cs_TYPE = ["MC" for _ in cs_ID]
        cs_ORIGIN = ["USERSPEC" for _ in cs_ID]
        cs_STATION = ["" for _ in cs_ID]

    channel = {"CH_ID": ch_ID, "CH_NDUSID": ch_NDUSID, "CH_NDDSID": ch_NDDSID}

    link = {
        "LK_ID": lk_ID,
        "LK_CMPSEQ": lk_CMPSEQ,
        "LK_NDUSID": lk_NDUSID,
        "LK_NDDSID": lk_NDDSID,
        "LK_RCUSID": lk_RCUSID,
        "LK_RCDSID": lk_RCDSID,
        "LK_TYPE": lk_TYPE,
    }

    nodes = {
        "ND_ID": nd_ID,
        "ND_FRMNO": nd_FRMNO,
        "ND_TYPE": nd_TYPE,
        "ND_XC": nd_XC,
        "ND_YC": nd_YC,
        "ND_DSID": nd_DSID,
        "ND_USID": nd_USID,
        "ND_US2ID": nd_US2ID,
        "ND_CSID": nd_CSID,
        "ND_RSID": nd_RSID,
        "ND_STID": nd_STID,
    }

    reach = {
        "RC_ID": rc_ID,
        "RC_NDUSID": rc_NDUSID,
        "RC_NDDSID": rc_NDDSID,
        "RC_ORDER": rc_ORDER,
        "RC_LENGTH": rc_LENGTH,
    }

    csec = {
        "CS_ID": cs_ID,
        "CS_NPTS": cs_NPTS,
        "CS_LOB": cs_LOB,
        "CS_LBT": cs_LBT,
        "CS_ROB": cs_ROB,
        "CS_RBT": cs_RBT,
        "CS_SVTYPE": cs_SVTYPE,
        "CS_TYPE": cs_TYPE,
        "CS_ORIGIN": cs_ORIGIN,
        "CS_STATION": cs_STATION,
    }

    csprf = {
        "CP_ID": cp_ID,
        "CP_CSID": cp_CSID,
        "CP_POSIDX": cp_POSIDX,
        "CP_W": cp_W,
        "CP_Z": cp_Z,
        "CP_RGH": cp_RGH,
        "CP_BLOCK": cp_BLOCK,
    }

    df_channel = pd.DataFrame.from_dict(channel)
    df_link = pd.DataFrame.from_dict(link)
    df_nodes = pd.DataFrame.from_dict(nodes)
    df_reach = pd.DataFrame.from_dict(reach)
    df_csec = pd.DataFrame.from_dict(csec)
    df_csprf = pd.DataFrame.from_dict(csprf)

    return df_nodes, df_channel, df_link, df_reach, df_csec, df_csprf, img_reach_asc_new


def write_cche1d_dat_file(filename, df):
    # This function automatically appends to the filename the appropriate
    # _channel.dat
    # _nodes.dat

    header_list = list(df.columns.values)

    if "CH_ID" in header_list:
        filename = filename + "_channel.dat"
    elif "ND_ID" in header_list:
        filename = filename + "_nodes.dat"
    elif "LK_ID" in header_list:
        filename = filename + "_link.dat"
    elif "RC_ID" in header_list:
        filename = filename + "_reach.dat"
    elif "CS_ID" in header_list:
        filename = filename + "_csec.dat"
    elif "CP_ID" in header_list:
        filename = filename + "_csprf.dat"
    else:
        filename = filename + ".dat"

    with open(filename, "w") as f:
        f.write(f"{df.shape[0]}\n")

    df.to_csv(filename, mode="a", sep="\t", index=False)

    print(f"{filename} successfully written")


def get_intermediate_nodes_img(usrowcol, dsrowcol, img_reach, rsize=1, distance=1):
    """
    This function returns the pixel coordinates in sequential order from usrowcol (tuple)
    to dsrowcol (tuple) in a form of a list of tuples.
        It is assumed that the path does not contain loops or branches, just an 8-connected path
    It is also assumed that all data provided is 0-indexed
        The returned path also includes the beginning and end nodes
    """

    img = np.copy(img_reach)

    rows, cols = np.nonzero(img)

    pixels_to_visit = [(r, c) for r, c in zip(rows, cols)]

    visited_pixels = [usrowcol]

    current_pixel = usrowcol
    cumulative_distance = [0]

    n_int_distance_traveled = 0  # Number of integer times "distance" was traveled

    if distance < rsize:
        distance = rsize  # Smallest resolution is the raster size

    # Possible directions. The order is important, the first four correspond to the immediate N,S,E,W to use before NE, NW, SW, SE...
    deltas = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]

    while pixels_to_visit:
        # Pop it out of the pixels_to_visit list
        pixels_to_visit.pop(pixels_to_visit.index(current_pixel))

        # Generate possible neighbors
        candidate_neighbors = [tuple(np.add(current_pixel, dl)) for dl in deltas]

        # Find which of those is in the pixels_to_visit
        candidates_matches = [c for c in candidate_neighbors if c in pixels_to_visit]

        if len(candidates_matches) == 0 and len(pixels_to_visit) != 0:
            raise Exception(
                "Could not find the rest of intermediate pixels, this should not happen, if you get this error something serious is going on"
            )
        elif len(candidates_matches) == 1:
            # Expected behavior

            dl = np.subtract(candidates_matches[0], current_pixel)
            if dl[0] != 0 and dl[1] != 0:
                factor = np.sqrt(2)  # Account for diagonal travel
            else:
                factor = 1

            current_pixel = candidates_matches[0]

            # n_int_distance_traveled = np.floor(cumulative_distance[-1]/distance) # Update integer number of distances traveled
            cumulative_distance.append(cumulative_distance[-1] + factor * rsize)

            if (
                np.floor(cumulative_distance[-1] / distance)
                == n_int_distance_traveled + 1
            ):
                visited_pixels.append(current_pixel)
                # print(f'Pixel kept,    cumdist = {cumulative_distance[-1]:.2f}, cumdist/dist = {cumulative_distance[-1]/distance:.2f}, n_int_distance_traveled+1 = {n_int_distance_traveled+1}, cumdist/dist int = {np.floor(cumulative_distance[-1]/distance)}, distance = {distance}, rsize = {rsize}')
            # else:
            # print(f'Pixel skipped, cumdist = {cumulative_distance[-1]:.2f}, cumdist/dist = {cumulative_distance[-1]/distance:.2f}, n_int_distance_traveled+1 = {n_int_distance_traveled+1}, cumdist/dist int = {np.floor(cumulative_distance[-1]/distance)}, distance = {distance}, rsize = {rsize}')

            n_int_distance_traveled = np.floor(cumulative_distance[-1] / distance)

        elif len(candidates_matches) > 1:
            # print("The shape of the path presents an ambiguous choice, the closest choice is taken")
            current_pixel = candidates_matches[0]
            visited_pixels.append(current_pixel)

    # Test if last pixel is the same as the provided end pixel
    if current_pixel != dsrowcol:
        raise Exception(
            "Mismatch with last found pixel and the expected end pixel, review data"
        )

    # Make sure dsrowcol is included in the visited_pixels
    if visited_pixels[-1] != dsrowcol:
        visited_pixels.append(dsrowcol)

    return visited_pixels


def collect_data_along_img_path(img, path, oneindexed=False):
    # img is a numpy array
    # path is a list of (row,col) index coordinates
    # data is the value in img for every (row,col) point

    if oneindexed:  # Index starts at 1 for the row/col coordinates
        data = [img[rc[0] - 1, rc[1] - 1] for rc in path]
    else:
        data = [img[rc[0], rc[1]] for rc in path]

    return data


def compute_cche1d_reaches_length(x, y):
    # x,y are two lists of length N containing the coordinates of each node in a channel "link" (in the CCHE1D understanding)
    # the function returns a rc_length list = [length_between_nodes(0,1), lenth_between_nodes(1,2),..., length_between_nodes(i,i+1), ... length_between_nodes(N-2,N-1)]

    rc_length = np.sqrt(
        (np.array(x[0:-1]) - np.array(x[1:])) ** 2
        + (np.array(y[0:-1]) - np.array(y[1:])) ** 2
    )

    return rc_length


def rowcol2latlon_esri_asc(geomatrix, row, col, oneindexed=False):
    # The provided row col NEED to be in 0-index. If oneindexed is provided (i.e. input assumes that the first row is row = 1
    # then an adjustment needs to be done
    if oneindexed:
        row -= 1
        col -= 1

    forwardtrans = Affine.from_gdal(*geomatrix)
    forwardtrans_cellcentered = forwardtrans * Affine.translation(0.5, 0.5)
    # forwardtrans_cellcentered = forwardtrans

    lon, lat = forwardtrans_cellcentered * (col, row)
    return (lat, lon)


def latlon2rowcol_esri_asc(geomatrix, lat, lon, oneindexed=False):
    forwardtrans = Affine.from_gdal(*geomatrix)
    forwardtrans_cellcentered = forwardtrans * Affine.translation(0.5, 0.5)
    reversetrans_cellcentered = ~forwardtrans_cellcentered
    col, row = reversetrans_cellcentered * (lon, lat)

    if oneindexed:
        row += 1
        col += 1

    return (int(row), int(col))


def read_esri_asc_file(filename, read_array=True):
    if isinstance(filename, Path):
        filename = str(filename.absolute())

    dataset = gdal.Open(filename)
    ncols = dataset.RasterXSize  # int
    nrows = dataset.RasterYSize  # int
    nodataval = dataset.GetRasterBand(1).GetNoDataValue()
    geoMatrix = dataset.GetGeoTransform()
    if read_array:
        img = dataset.ReadAsArray()
    else:
        img = None
    return img, geoMatrix, ncols, nrows, nodataval, dataset


def visualize_cche1d_nodes(df, show=True, renderer="notebook", text="ND_ID"):
    # fig = px.scatter(df, x="ND_XC", y="ND_YC", text="ND_FRMNO")
    dfcopy = df.copy()
    dfcopy["ND_TYPE_SHOW"] = df["ND_TYPE"].astype(str)
    # fig = px.scatter(df, x="ND_XC", y="ND_YC", color=color, text=what, title=what)
    fig = px.scatter(
        dfcopy,
        x="ND_XC",
        y="ND_YC",
        symbol=dfcopy["ND_TYPE"],
        text=text,
        color="ND_TYPE_SHOW",
        title="ND_FRMNO",
        hover_name="ND_ID",
        hover_data=["ND_USID", "ND_DSID", "ND_US2ID", "ND_CSID"],
    )
    fig.update_traces(textposition="bottom left", marker={"size": 5})
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=900, showlegend=False)

    if show:
        fig.show(renderer=renderer)
    return fig


def visualize_strahler_number(
    img, geomatrix=None, show=True, title="Strahler Number", renderer="notebook"
):
    # img : 2D-Numpy array

    nrows, ncols = img.shape

    cols = list(range(1, ncols + 1))
    rows = list(range(1, nrows + 1))

    imgcopy = img.copy()
    imgcopy = np.where(imgcopy == 0, np.nan, imgcopy)

    if geomatrix == None:
        x = cols
        y = rows
    elif len(geomatrix) == 6:
        y = [rowcol2latlon_esri_asc(geomatrix, r, 1, oneindexed=True) for r in rows]
        y, _ = zip(*y)
        y = list(y)

        x = [rowcol2latlon_esri_asc(geomatrix, 1, c, oneindexed=True) for c in cols]
        _, x = zip(*x)
        x = list(x)
    else:
        raise Exception("Incorrect number of arguments in the geomatrix tuple")

    # fig = px.imshow(img, x=x, y=y, origin='lower', text_auto=True, color_continuous_scale=px.colors.qualitative.D3)

    data = go.Heatmap(
        z=imgcopy,
        x=x,
        y=y,
        #   text=imgcopy,
        #   texttemplate="%{text:1f}",
        colorscale=px.colors.qualitative.D3,
        showscale=False,
        hoverongaps=False,
        hovertemplate="<extra></extra> x: %{x:.3f} <br /> y: %{y:.3f} <br /> Strahler Number: %{z:1.f}",
    )
    fig = go.Figure(data=data)

    fig.update_layout(
        title=title, xaxis_title="Easting (m)", yaxis_title="Northing (m)"
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if show:
        fig.show(renderer=renderer)

    return fig


def visualize_reaches_id(
    img, geomatrix=None, show=True, title="Reach IDs", renderer="notebook"
):
    # img : 2D-Numpy array

    nrows, ncols = img.shape

    cols = list(range(1, ncols + 1))
    rows = list(range(1, nrows + 1))

    imgcopy = img.copy()
    imgcopy = np.where(imgcopy == 0, np.nan, imgcopy)

    if geomatrix == None:
        x = cols
        y = rows
    elif len(geomatrix) == 6:
        y = [rowcol2latlon_esri_asc(geomatrix, r, 1, oneindexed=True) for r in rows]
        y, _ = zip(*y)
        y = list(y)

        x = [rowcol2latlon_esri_asc(geomatrix, 1, c, oneindexed=True) for c in cols]
        _, x = zip(*x)
        x = list(x)
    else:
        raise Exception("Incorrect number of arguments in the geomatrix tuple")

    # fig = px.imshow(img, x=x, y=y, origin='lower', text_auto=True, color_continuous_scale=px.colors.qualitative.D3)

    data = go.Heatmap(
        z=imgcopy,
        x=x,
        y=y,
        #   text=imgcopy,
        #   texttemplate="%{text:1f}",
        colorscale=px.colors.qualitative.D3,
        showscale=False,
        hoverongaps=False,
        hovertemplate="<extra></extra> x: %{x:.3f} <br /> y: %{y:.3f} <br /> Reach: %{z:1.f}",
    )
    fig = go.Figure(data=data)

    fig.update_layout(
        title=title, xaxis_title="Easting (m)", yaxis_title="Northing (m)"
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    if show:
        fig.show(renderer=renderer)

    return fig
