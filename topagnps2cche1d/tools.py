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

def reorder_topagnps_reaches_into_cche1d_links(img, rowcol_outlet_tuple, nodataval=0, ondeindex=False):
    
    
    nodataval = int(nodataval)
    nodataval = np.array(nodataval)
    topaz_reaches = np.unique(img) # ignoring the first one being zeros
    topaz_reaches = np.setdiff1d(topaz_reaches,nodataval)
    topaz_img = np.full_like(img, nodataval)

    # Find reach number 

    for treach_id in topaz_reaches:
        reach_img = np.where(img==treach_id,1,0)

    pass

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