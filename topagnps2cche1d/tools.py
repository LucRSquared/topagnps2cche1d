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

def build_network(dfagflow, img_flovec, root=None):

    # For now, nothing is done if more than 2 inflows come at a junction
    # If root is specified then all the downstream reaches are ignored

    network = {}

    # All Reaches that receive a tributary in the network
    receiving_reaches = np.unique(dfagflow['Receiving_Reach'].to_numpy())

    # Build the entire network
    for rr in receiving_reaches:
        receiving_reach = dfagflow.loc[dfagflow['Reach_ID'] == rr,:]
        juncrowcol = tuple(receiving_reach[['Upstream_End_Row','Upstream_End_Column']].to_records(index=False)[0])

        candidates = dfagflow.loc[(dfagflow['Receiving_Reach']==rr) & \
                                (dfagflow['Reach_ID']!=rr),:]
                          
        network[rr] = get_counterclockwise_inflows(candidates, juncrowcol, img_flovec, oneindexed=True)

    
    if root is not None:
        reaches_to_keep = dfs_iterative_postorder(network, root)

        new_network = {}

        for r in reaches_to_keep:
            if r in network.keys():
                new_network[r] = network[r]

        return new_network

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

def convert_topagnps_output_to_cche1d_input(filepath_agflow, filepath_flovec, filepath_annagnps_reach_ids, outfilespath=None, writefiles = False):
    # This script takes topagns reaches and changes the numbering according to the CCHE1D numbering
    # system for links
    # ASSUMPTIONS :
    # - Only 2 inflows per junction (TopAGNPS guarantees it but AnnAGNPS is more liberal)
    # - The Outlet "reach" is its own reach but has 0 length, potential problem !!!!

    # Inputs : 
    # - FILEPATH to Agflow file
    # - FILEPATH to FLOVEC.ASC file
    # -
    # - outfilespath: path to output dat files
    # - writefiles: T/F

    # Outputs:
    # - cche1d_reordered_reaches giving the new order of previously ordered reaches from 1 to N_max_reaches

    if outfilespath is None:
        outfilespath = os.getcwd()

    # Reading
    img_flovec = read_esri_asc_file(filepath_flovec)[0]
    img_reach_asc, geomatrix, _, _, _, _ = read_esri_asc_file(filepath_annagnps_reach_ids)
    dfagflow = read_agflow_reach_data(filepath_agflow)

    # # Get Outlet ROW/COL "reach" id (it's the one that has zero reach length)
    # # outlet_reach_id = dfagflow.loc[dfagflow['Reach_Length_[m]']==0,'Reach_ID'].values[0]

    # # !!! Potential Problem with outlet reach
    # # initial_outlet_reach_index = dfagflow.index[dfagflow['Reach_Length_[m]']==0].values[0]
    # # dfagflow.drop(labels=initial_outlet_reach_index, axis=0, inplace=True)

    outlet_reach_id = dfagflow.loc[(dfagflow['Distance_Downstream_End_to_Outlet_[m]']==0) & (dfagflow['Reach_Length_[m]']!=0),'Reach_ID'].values[0]


    # Build the flow network (each node points to the counterclock wise list of tributaries at the upstream junction)
    network = build_network(dfagflow, img_flovec, root=outlet_reach_id)

    cche1d_reordered_reaches = dfs_iterative_postorder(network, outlet_reach_id)

    reordered_network = reorder_network(network, cche1d_reordered_reaches)

    df_nodes, df_channel, df_link = create_cche1d_nodes_and_channel_tables(dfagflow, geomatrix, img_reach_asc, cche1d_reordered_reaches, reordered_network)


    return df_nodes, df_channel, df_link

def apply_permutation_int_array(original_arr, permutation_vect):

    renumbered_arr = np.copy(original_arr)

    # Delete entries that aren't in the permutation_vect
    unique_vals = np.unique(renumbered_arr)
    vals_to_replace_by_zero = set(unique_vals) - set(permutation_vect)

    for v in vals_to_replace_by_zero:
        renumbered_arr[renumbered_arr==v] = 0

    for idx, newid in enumerate(permutation_vect,1):
        renumbered_arr[original_arr==idx] = newid

    return renumbered_arr

def apply_permutation_int_dfagflow(dfagflow, permutation_vect):

    # dfagflow_new = dfagflow.copy(deep=True)
    # Keep only the reaches present in the permutation vector
    dfagflow_new = dfagflow[dfagflow['Reach_ID'].isin(permutation_vect)].copy()

    dfagflow_new['New_Reach_ID'] = dfagflow_new.apply(lambda x: permutation_vect.index(int(x['Reach_ID']))+1, axis=1)
    dfagflow_new['New_Receiving_Reach'] = dfagflow_new.apply(lambda x: permutation_vect.index(int(x['Reach_ID']))+1, axis=1)

    return dfagflow_new

def create_cche1d_nodes_and_channel_tables(dfagflow, geomatrix, img_reach_asc, permutation_vect, reordered_network):

    # INPUTS:
    # - dfagflow: Original AgFlow dataframe
    # - img_reach_asc: Array with pixels numbered according to the old (topagnps) numbering of reaches
    # - permuation_vect: permutation vector for new numbering of reaches
    # - outfilepath: path to output dat file
    # - writefile: T/F
    #
    # OUTPUTS:
    # - df_nodes table
    # - df_channel table
    # - df_link table

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

    # Initialize Channel Arrays
    ch_ID = []
    ch_NDUSID = []
    ch_NDDSID = []

    # Initialize Link Arrays
    lk_ID = []
    lk_CMPSEQ = []
    lk_NDUSID = []
    lk_NDDSID = []
    lk_RCUSID = []
    lk_RCDSID = []
    # lk_TYPE = []

    # Useful indexing tools
    FirstInflowLastNodeIDForSecondInflow = {}
    FirstInflowLastNodeIDForReceivingReach = {}
    SecondInflowLastNodeIDForReceivingReach = {}

    FirstInflowLastNodeAbsoluteIndexForReceivingReach = {}
    SecondInflowLastNodeAbsoluteIndexForReceivingReach = {}

    N_max_reach = dfagflow_new['New_Reach_ID'].max()

    first_inflows = [e[0] for e in reordered_network.values()]
    second_inflows = [e[1] for e in reordered_network.values()]
    list_of_receiving_reaches = [k for k in reordered_network.keys()]

    # outlet_reach = dfagflow_new.loc[dfagflow_new['Reach_Length_[m]']==0,['New_Reach_ID']]
    outlet_reach_id = dfagflow_new.loc[(dfagflow_new['Distance_Downstream_End_to_Outlet_[m]']==0) & (dfagflow_new['Reach_Length_[m]']!=0) ,'New_Reach_ID'].values[0]

    nd_counter = 1

    # Goes from 1 to N_max_reach
    for reach_id in range(1,N_max_reach+1):

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

        # Channel table
        ch_ID.append(reach_id)
        ch_NDUSID.append(nd_ID_tmp[0])
        ch_NDDSID.append(nd_ID_tmp[-1])

        # Link table
        lk_ID.append(reach_id)
        lk_CMPSEQ.append(reach_id) # the loop already goes in the computational sequence
        lk_NDUSID.append(reach_id)
        lk_NDDSID.append(reach_id)
        lk_RCUSID.append(reach_id)
        lk_RCDSID.append(reach_id)

        
        nd_CSID_tmp = nd_ID_tmp.copy()        
        nd_FRMNO_tmp = nd_ID_tmp.copy() # Those IDs are the same because this procedure is done in the computational order
        nd_US2ID_tmp = [-1 for _ in nd_ID_tmp]
        nd_TYPE_tmp = [6 for _ in nd_ID_tmp] # assign value 6 by default, special cases are treated hereafter

        

        if reach_id == outlet_reach_id:
            nd_TYPE_tmp[-1] = 9 # Label last node as the outflow
            nd_DSID_tmp.append(nd_DSID_tmp[-1]) # The "downstream" node of the outlet node is itself
        else:
            nd_TYPE_tmp[-1] = 3 # Last node is end of link by default
            
            # Complete in the main array the DS node id of the correponding inflows
            idx_firstinflow_last_node = FirstInflowLastNodeAbsoluteIndexForReceivingReach[reach_id][0]
            id_ds =  FirstInflowLastNodeAbsoluteIndexForReceivingReach[reach_id][1]
            nd_DSID[idx_firstinflow_last_node] = id_ds

            idx_second_inflow_last_node = FirstInflowLastNodeAbsoluteIndexForReceivingReach[reach_id][0]
            id_ds =  SecondInflowLastNodeAbsoluteIndexForReceivingReach[reach_id][1]
            nd_DSID[idx_second_inflow_last_node] = id_ds

            nd_DSID_tmp = nd_ID_tmp[1:]
            nd_DSID_tmp.append(None) # Leave a placeholder (to keep the index correct)
                                     # until it can be resolved when the algorithm 
                                     # gets to the receiving reach

        # this can be consolidated down there
        if reach_id not in list_of_receiving_reaches:
            nd_USID_tmp = [-1] # First node is a source node 
            nd_USID_tmp.extend(nd_ID_tmp[0:-1])

            nd_TYPE_tmp[0] = 0 # First node is a source node

        else:
            nd_USID_tmp = [SecondInflowLastNodeIDForReceivingReach[reach_id]]
            nd_USID_tmp.extend(nd_ID_tmp[0:-1])
            nd_US2ID_tmp[0] = [FirstInflowLastNodeIDForReceivingReach[reach_id]]

            nd_CSID_tmp[0] = FirstInflowLastNodeIDForReceivingReach[reach_id]

            nd_TYPE_tmp[0] = 2 # First node is a junction node


        if reach_id in first_inflows:
            idx1 = first_inflows.index(reach_id)
            FirstInflowLastNodeIDForSecondInflow[second_inflows[idx1]] = nd_ID_tmp[-1]
            FirstInflowLastNodeIDForReceivingReach[list_of_receiving_reaches[idx1]] = nd_ID_tmp[-1]
            FirstInflowLastNodeAbsoluteIndexForReceivingReach[list_of_receiving_reaches[idx1]] = (len(nd_ID)+len(nd_ID_tmp)-1, nd_ID_tmp[0])

        if reach_id in second_inflows:
            idx2 = second_inflows.index(reach_id)
            SecondInflowLastNodeIDForReceivingReach[list_of_receiving_reaches[idx2]] = nd_ID_tmp[-1]
            SecondInflowLastNodeAbsoluteIndexForReceivingReach[list_of_receiving_reaches[idx1]] = (len(nd_ID)+len(nd_ID_tmp)-1, nd_ID_tmp[0])
            nd_CSID_tmp[-1] = FirstInflowLastNodeIDForSecondInflow[reach_id]

        

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
        nd_US2ID.extend(nd_US2ID_tmp)


    nd_RSID = [-1 for _ in nd_ID]
    nd_STID = [1 for _ in nd_ID]

    lk_TYPE = [1 for _ in ch_ID]

    channel = {'CH_ID': ch_ID, 'CH_NDUSID': ch_NDUSID, 'CH_NDDSID': ch_NDDSID}
    
    link = {'LK_ID': lk_ID,
            'LK_CMPSEQ': lk_CMPSEQ,
            'LK_NDUSID': lk_NDUSID,
            'LK_NDDSID': lk_NDDSID,
            'LK_RCUSID': lk_RCUSID,
            'LK_RCDSID': lk_RCDSID,
            'LK_TYPE': lk_TYPE}

    nodes = {'ND_ID': nd_ID,
             'ND_FRMNO': nd_FRMNO,
             'ND_TYPE': nd_TYPE,
             'ND_XC': nd_XC,
             'ND_YC': nd_YC,
             'ND_DSID': nd_DSID,
             'ND_USID': nd_USID,
             'ND_US2ID': nd_US2ID,
             'ND_CSID': nd_CSID,
             'ND_RSID': nd_RSID,
             'ND_STID': nd_STID}

    df_channel = pd.DataFrame.from_dict(channel)
    df_link = pd.DataFrame.from_dict(link)
    df_nodes = pd.DataFrame.from_dict(nodes)

    return df_nodes, df_channel, df_link

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