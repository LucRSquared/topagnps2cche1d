import csv
import pandas as pd
import plotly.express as px

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

def visualize_cche1d_nodes(df,show=True):
    fig = px.scatter(df, x="ND_XC", y="ND_YC", text="ND_FRMNO")
    fig.update_traces(textposition='bottom left')
    if show==True:
        fig.show()
    return fig