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

# def reorder_topagnps_reaches_into_cche1d_links(img, rowcol_outlet_tuple, nodataval=0, ondeindex=False):
    
    
#     nodataval = int(nodataval)
#     nodataval = np.array(nodataval)
#     topaz_reaches = np.unique(img) # ignoring the first one being zeros
#     topaz_reaches = np.setdiff1d(topaz_reaches,nodataval)
#     topaz_img = np.full_like(img, nodataval)

#     # Find reach number 

#     for treach_id in topaz_reaches:
#         reach_img = np.where(img==treach_id,1,0)

#     pass

#!!! DO something if more than two reaches come up?
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

def get_counterclockwise_inflows(candidates, juncrowcol, img_flovec, oneindexed=False):

    # WARNING The Assumption is made that the orientation of the plane is indirect
    # due to ij orientation j <-> x / i <-> j

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
    # - The Outlet is its own reach

    # Inputs : 
    # - FILEPATH of DatFrame of Agflow file
    # - FILEPATH of FLOVEC.ASC file

    # Reading
    img_flovec = read_esri_asc_file(filepath_flovec)[0]
    dfagflow = read_agflow_reach_data(filepath_dfagflow)

    # Get Outlet ROW/COL "reach" id (it's the one that has zero reach length)
    outlet_reach_id = dfagflow.loc[dfagflow['Reach_Length_[m]']==0,'Reach_ID'].values[0]

    # Build the flow network (each node points to the counterclock wise list of tributaries at the upstream junction)
    network = build_network(dfagflow, img_flovec)

    cche1d_reordered_reaches = dfs_iterative_postorder(network,outlet_reach_id)

    return cche1d_reordered_reaches

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

# def get_raster_img_asc_rowcol_val(geomatrix, row, col, img, oneindexed=False):


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