import os
import csv
import pandas as pd
import numpy as np
import plotly.express as px
from osgeo import gdal
from affine import Affine


def read_first_line(filename):
    with open(filename, 'r') as f:
        r = csv.reader(f)
        num = next(r)
        num = int(num[0])
    return num

def read_cche1d_dat_file(filename):
    numlines = read_first_line(filename)
    df = pd.read_csv(filename, skiprows=1, nrows=numlines, sep='\t', header='infer')
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

# the nodes file is slightly different
def read_cche1d_nodes_dat(filename):
    numlines = read_first_line(filename)
    df = pd.read_csv(filename, skiprows=1, nrows=numlines, sep='\t', header='infer',\
        index_col=False)
    df.reset_index(drop=True, inplace=True)
    try:
        df.drop(labels='Unnamed: 11',axis=1,inplace=True)
    except:
        pass
    return df

def read_agflow_reach_data(filename):
    df = pd.read_csv(filename, header='infer', index_col=False)
    df.reset_index(drop=True, inplace=True)
    return df

def find_extremities_binary(img):
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(img)

    # Get length of the path by counting the pixels
    skel_length = len(rows)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):

        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')

        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = img[row_neigh,col_neigh].ravel() != 0

        # If the number of non-zero locations equals 2, add this to 
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2 :
            skel_coords.append((r,c))
        elif np.sum(pix_neighbourhood) == 1: # this is a one node reach
            skel_coords.append((r,c))


    return skel_coords, skel_length

def build_network(dfagflow, img_flovec):

    # For now, nothing is done if more than 2 inflows come at a junction

    network = {}

    # All Reaches that receive a tributary in the network
    receiving_reaches = np.unique(dfagflow['Receiving_Reach'].to_numpy())

    for rr in receiving_reaches:
        receiving_reach = dfagflow.loc[dfagflow['Reach_ID'] == rr,:]
        juncrowcol = tuple(receiving_reach[['Upstream_End_Row','Upstream_End_Column']].to_records(index=False)[0])

        candidates = dfagflow.loc[(dfagflow['Receiving_Reach']==rr) & \
                                (dfagflow['Reach_ID']!=rr),:]
                          
        network[rr] = get_counterclockwise_inflows(candidates, juncrowcol, img_flovec, oneindexed=True)

    return network

def reorder_network(network, permutation_vect):

    reordered_network = {}

    for k in list(network):
        reordered_network[permutation_vect.index(k)+1] = [permutation_vect.index(e)+1 for e in network.pop(k)]

    return reordered_network

def get_counterclockwise_inflows(candidates, juncrowcol, img_flovec, oneindexed=False):

    # WARNING The Assumption is made that the orientation of the plane is indirect
    # due to ij orientation j <-> x / i <-> y

    # Candidates is a subselection of the dfagflow dataframe with the list of candidates
    # juncrowcol is a tuple of the upstream (row,col) coordinates of the receiving reach
    
    cand_reach_ids = [rid for rid in candidates['Reach_ID']]

    DS_end = [x+y*1j for x, y in zip(candidates['Downstream_End_Column'], 
                                     candidates['Downstream_End_Row'])]
    
    if not oneindexed:
        juncrowcol = np.add(juncrowcol,(1,1))

    junc_complex = juncrowcol[1] + juncrowcol[0]*1j

    # Get flovec of outrowcol
    if oneindexed:
        idx_flovec = int(img_flovec[juncrowcol[0]-1,juncrowcol[1]-1])
    else:
        idx_flovec = int(img_flovec[juncrowcol[0],juncrowcol[1]])


    refvec_flovec = {1:-1-1j,
                    2:-1j,
                    3:1-1j,
                    4:-1+0j,
                    6:1+0j,
                    7:-1-1j,
                    8:+1j,
                    9:1+1j}
    # These numbers are so to represent the direction of the flow
    # in a cell because the orientation of the plane is ij not xy
    # See TOPAZ User Manual FLOVEC description

    dsends_minus_junc_row_col = [(ds-junc_complex)/refvec_flovec[idx_flovec] for ds in DS_end]

    angles_of_incoming_reaches = np.angle(dsends_minus_junc_row_col, deg=True)
    # Make sure the angles are in the [0, 360 [ range
    angles_of_incoming_reaches = (angles_of_incoming_reaches+360) % 360


    # Same here, the descending order is taken because of the ij orientation of the plane
    idxsort = np.argsort(angles_of_incoming_reaches)
    idxsort = idxsort[::-1]

    reaches_counterclockwise_order = list(np.array(cand_reach_ids)[idxsort])

    return reaches_counterclockwise_order

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

    if outlet_reach not in network:
        return []

    stack = [outlet_reach]
    result = []

    while stack:
        reach = stack.pop()
        result.append(reach)

        if reach not in network: # if reach has inflows
            inflows = []
        else:
            inflows = network[reach]

        for tributaries in inflows:
            stack.append(tributaries)

    postorder = result[::-1]

    return postorder

def dfs_recursive_postorder(network, reach, postorder=[]):

    # Depth First Search Post Order Tree Traversal (Recursive)

    # network is a dict
    # Each entry in the network is a reach and points to the list of inflow reaches ordered from "right to left"
    #
    # Example: 
    # network = {0:[1,2], 1:[3,4], 2:[5,6], 6:[7]}
    # start = 0
    # postorder = dfs_recursive_postorder(network, start)

    if reach not in network:
        postorder.append(reach)
        return postorder

    inflows = network[reach]

    for c in inflows:
        dfs_recursive_postorder(network,c,postorder)
            
    postorder.append(reach)

    return postorder

def reorder_topagns_reaches_into_cche1d_link_agflow_method(filepath_dfagflow, filepath_flovec):
    # This script takes topagns reaches and changes the numbering according to the CCHE1D numbering
    # system for links
    # ASSUMPTIONS :
    # - Only 2 inflows per junction (TopAGNPS guarantees it but AnnAGNPS is more liberal)
    # - The Outlet "reach" is its own reach but has 0 length, potential problem !!!!

    # Inputs : 
    # - FILEPATH of DatFrame of Agflow file
    # - FILEPATH of FLOVEC.ASC file

    # Outputs:
    # - cche1d_reordered_reaches giving the new order of previously ordered reaches from 1 to N_max_reaches

    # Reading
    img_flovec = read_esri_asc_file(filepath_flovec)[0]
    dfagflow = read_agflow_reach_data(filepath_dfagflow)

    # Get Outlet ROW/COL "reach" id (it's the one that has zero reach length)
    outlet_reach_id = dfagflow.loc[dfagflow['Reach_Length_[m]']==0,'Reach_ID'].values[0]

    # !!! Potential Problem with outlet reach
    # initial_outlet_reach_index = dfagflow.index[dfagflow['Reach_Length_[m]']==0].values[0]
    # dfagflow.drop(labels=initial_outlet_reach_index, axis=0, inplace=True)

    # outlet_reach_id = dfagflow.loc[dfagflow['Distance_Downstream_End_to_Outlet_[m]']==0,'Reach_ID'].values[0]


    # Build the flow network (each node points to the counterclock wise list of tributaries at the upstream junction)
    network = build_network(dfagflow, img_flovec)

    cche1d_reordered_reaches = dfs_iterative_postorder(network, outlet_reach_id)

    reordered_network = reorder_network(network, cche1d_reordered_reaches)

    return cche1d_reordered_reaches

def apply_permutation_int_array(original_arr, permutation_vect):

    renumbered_arr = np.copy(original_arr)

    for idx, newid in enumerate(permutation_vect,1):
        renumbered_arr[original_arr==idx] = newid

    return renumbered_arr

def apply_permutation_int_dfagflow(dfagflow, permutation_vect):

    dfagflow_new = dfagflow.copy(deep=True)

    dfagflow_new['New_Reach_ID'] = dfagflow_new.apply(lambda x: permutation_vect[int(x['Reach_ID'])-1], axis=1)
    dfagflow_new['New_Receiving_Reach'] = dfagflow_new.apply(lambda x: permutation_vect[int(x['Receiving_Reach'])-1], axis=1)

    return dfagflow_new

def create_cche1d_nodes_and_channel_tables(dfagflow, geomatrix, img_reach_asc, permutation_vect, reordered_network, outfilepath=None, writefile = False):

    # INPUTS:
    # - dfagflow: Original AgFlow dataframe
    # - img_reach_asc: Array with pixels numbered according to the old (topagnps) numbering of reaches
    # - permuation_vect: permutation vector for new numbering of reaches
    # - outfilepath: path to output dat file
    # - writefile: T/F
    #
    # OUTPUTS:
    # - dfcche1d_nodes table
    # 

    # If unspecified, the output filepath is set as the current working directory
    if outfilepath is None:
        outfilepath = os.getcwd()

    # Apply new numbering to AgFlow Dataframe and img_reach_asc
    dfagflow_new = apply_permutation_int_dfagflow(dfagflow, permutation_vect)
    img_reach_asc_new = apply_permutation_int_array(img_reach_asc, permutation_vect)

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
    # nd_RSID = [] # Those are generated on their own later, identically -1
    # nd_STID = []

    # Initialize Channel Arrays
    ch_ID = []
    ch_NDUSID = []
    ch_NDDSID = []

    us_nd_ID_mem = {}
    ds_nd_ID_mem = {}

    N_max_reach = dfagflow_new['New_Reach_ID'].max()

    # list_of_receiving_reaches = np.unique(dfagflow['New_Receiving_Reach'])

    first_inflows = [e[0] for e in reordered_network.values()]
    second_inflows = [e[1] for e in reordered_network.values()]
    list_of_receiving_reaches = [k for k in reordered_network.keys()]

    # outlet_reach = dfagflow_new.loc[dfagflow_new['Reach_Length_[m]']==0,['New_Reach_ID']]
    outlet_reach_id = dfagflow.loc[dfagflow['Distance_Downstream_End_to_Outlet_[m]']==0,'New_Reach_ID'].values[0]

    nd_counter = 1

    # Goes from 1 to N_max_reach-1, the "last" reach is the outlet defined by TopAGNPS
    # but does not have any length and the penultimate reache's last pixel matches that point
    for reach_id in range(1,N_max_reach):

        # Get Upstream ROW/COL of current reach
        us_row, us_col = dfagflow_new.loc[dfagflow_new['New_Reach_ID']==reach_id,['Upstream_End_Row','Upstream_End_Column']].values[0]

        # Get Downstream ROW/COL of current reach
        ds_row, ds_col = dfagflow_new.loc[dfagflow_new['New_Reach_ID']==reach_id,['Downstream_End_Row','Downstream_End_Column']].values[0]

        # Keep only the pixels pertaining to the current reach
        curr_img_reach = np.where(img_reach_asc_new==reach_id,1,0)

        # Write function get_intermediate_nodes
        ordered_path = get_intermediate_nodes_img((us_row,us_col), (ds_row,ds_col), curr_img_reach)

        # Computing corresponding coordinates
        XCYC_tmp = [rowcol2latlon_esri_asc(geomatrix, rowcol[0], rowcol[1], oneindexed=True) for rowcol in ordered_path]
        XC_tmp, YC_tmp = zip(*XCYC_tmp)
        
        # For all reaches that are not the outlet
        if reach_id != outlet_reach_id:
            # Get receiving reach's upstream node (that will be the junction)
            receiving_reach_id = dfagflow_new.loc[dfagflow_new['New_Reach_ID']==reach_id,['New_Receiving_Reach']].values[0]
            junc_node_row, junc_node_col = dfagflow_new.loc[dfagflow_new['New_Reach_ID']==receiving_reach_id, ['Upstream_End_Row','Upstream_End_Column']]
            junc_node_x, junc_node_y = rowcol2latlon_esri_asc(geomatrix, junc_node_row, junc_node_col, oneindexed=True)

            # Append to the reach
            XC_tmp.append(junc_node_x)
            YC_tmp.append(junc_node_y)

        numpoints = len(XC_tmp)
        
        nd_ID_tmp = list(range(nd_counter,numpoints+1))

        # nd_DSID_tmp = nd_ID_tmp[1:]

        # this can be consolidated down there
        if reach_id not in list_of_receiving_reaches:
            nd_USID_tmp = [-1]
            nd_USID_tmp.extend(nd_ID_tmp[0:-1])
        else:
            pass
            # more complicated

        nd_CSID_tmp = nd_ID_tmp.copy()

        if reach_id in first_inflows:
            idx = first_inflows.index(reach_id)
            ds_nd_ID_mem[second_inflows[idx]] = nd_ID_tmp[-1]
            us_nd_ID_mem[list_of_receiving_reaches[idx]] = nd_ID_tmp[-1]

        if reach_id in second_inflows:
            nd_CSID_tmp[-1] = ds_nd_ID_mem[reach_id]

        if reach_id in list_of_receiving_reaches:
            nd_CSID_tmp[0] = us_nd_ID_mem[reach_id]
 

        nd_FRMNO_tmp = nd_ID_tmp.copy() # Those IDs are the same because this procedure is done in the computational order

        nd_TYPE_tmp = [6 for _ in nd_ID_tmp] # assign value 6 by default, special cases are treated hereafter

        # Increment nd_counter
        nd_counter += numpoints 

        # Deal with nd_DSID, nd_USID, nd_US2ID
        nd_DSID_tmp = nd_ID_tmp.copy()
        nd_USID_tmp = nd_ID_tmp.copy()
        nd_US2ID_tmp = nd_ID_tmp.copy()

        if reach_id not in list_of_receiving_reaches:
            # This is a reach with a source node
            nd_TYPE_tmp[0] = 0 # First node is a source node
            nd_TYPE_tmp[-1] = 3 # Last node is end of link by default

        if (us_row, us_col) in junction_nodes:
            nd_TYPE_tmp[0] = 2
            # reset junction nodes list
            junction_nodes = []

        if reach_id == outlet_reach_id:
            nd_TYPE_tmp[-1] = 9 # Label last node as the outflow

        # Append all the temporary lists
        nd_ID.extend(nd_ID_tmp)
        nd_FRMNO.extend(nd_FRMNO_tmp)
        nd_TYPE.extend(nd_TYPE_tmp)
        nd_XC.extend(XC_tmp)
        nd_YC.extend(YC_tmp)
        nd_CSID.extend(nd_CSID_tmp)

    nd_RSID = [-1 for _ in range(1,len(nd_ID)+1)]
    nd_STID = nd_RSID.copy()

    return

def get_intermediate_nodes_img(usrowcol, dsrowcol, img_reach):
    # This function returns the pixel coordinates in sequential order from usrowcol (tuple)
    # to dsrowcol (tuple) in a form of a list of tuples.
    #
    # It is assumed that the path does not contain loops or branches, just an 8-connected path

    img = np.copy(img_reach)

    rows, cols = np.nonzero(img)

    pixels_to_visit = [(r, c) for r, c in zip(rows,cols)]

    visited_pixels = [usrowcol]

    current_pixel = usrowcol

    # Possible directions. The order is important, the first four correspond to the immediate N,S,E,W to use before NE, NW, SW, SE...
    deltas = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]

    while pixels_to_visit:

        # Pop it out of the pixels_to_visit list
        pixels_to_visit.pop(pixels_to_visit.index(current_pixel))

        # Generate possible neighbors
        candidate_neighbors = [tuple(np.add(current_pixel,dl)) for dl in deltas]

        # Find which of those is in the pixels_to_visit
        candidates_matches = [c for c in candidate_neighbors if c in pixels_to_visit]

        if len(candidates_matches) == 0 and len(pixels_to_visit) != 0:
            raise Exception("Could not find the rest of intermediate pixels, this should not happen, if you get this error something serious is going on")
        elif len(candidates_matches) == 1:
            # Expected behavior 
            current_pixel = candidates_matches[0]
            visited_pixels.append(current_pixel)
            
        elif len(candidates_matches) > 1:
            # print("The shape of the path presents an ambiguous choice, the closest choice is taken")
            current_pixel = candidates_matches[0]
            visited_pixels.append(current_pixel)
    
    # Test if last pixel is the same as the provided end pixel
    if current_pixel != dsrowcol:
        raise Exception('Mismatch with last found pixel and the expected end pixel, review data')

    return visited_pixels

def rowcol2latlon_esri_asc(geomatrix, row, col, oneindexed=False):
    
    if oneindexed:
        row -= 1
        col -= 1

    forwardtrans = Affine.from_gdal(*geomatrix)
    forwardtrans_cellcentered = forwardtrans * Affine.translation(0.5,0.5)
    # forwardtrans_cellcentered = forwardtrans

    lon, lat = forwardtrans_cellcentered * (col, row)
    return (lat, lon)

def latlon2rowcol_esri_asc(geomatrix, lat, lon, oneindexed=False):
    forwardtrans = Affine.from_gdal(*geomatrix)
    forwardtrans_cellcentered = forwardtrans * Affine.translation(0.5,0.5)
    reversetrans_cellcentered = ~forwardtrans_cellcentered
    col, row = reversetrans_cellcentered * (lon, lat)

    if oneindexed:
        row += 1
        col += 1

    return (int(row), int(col))

def read_esri_asc_file(filename):
    dataset = gdal.Open(filename)
    ncols = dataset.RasterXSize # int
    nrows = dataset.RasterYSize # int
    nodataval = dataset.GetRasterBand(1).GetNoDataValue()
    geoMatrix = dataset.GetGeoTransform()
    img = dataset.ReadAsArray()
    return img, geoMatrix, ncols, nrows, nodataval, dataset

def visualize_cche1d_nodes(df, show=True, what="ND_ID", color="ND_TYPE"):
    # fig = px.scatter(df, x="ND_XC", y="ND_YC", text="ND_FRMNO")
    df["ND_TYPE"] = df["ND_TYPE"].astype(str)
    fig = px.scatter(df, x="ND_XC", y="ND_YC", color=color, text=what, title=what)
    fig.update_traces(textposition='bottom left')
    if show==True:
        fig.show()
    return fig