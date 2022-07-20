import topagnps2cche1d.tools as t2c

# Location of all relevant files:
srcfolder = 'topagnps_ohio_files/'
filepath_annagnps_reach_ids = srcfolder + 'AnnAGNPS_Reach_IDs.asc'
filepath_agflow = srcfolder + 'AgFlow_Reach_Data.csv'
filepath_flovec = srcfolder + 'FLOVEC.asc'
filepath_netw = srcfolder + 'NETW.asc'

min_strahler = 2
distance = 20

default_xsection = {'type' : 'default',
                    'CP_Ws': [-43, -35, -13, -10, 10, 13, 35, 43],
                    'CP_Zs': [6, 2, 2, 0, 0, 2, 2, 6],
                    'CP_RGHs': [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]}

outputfolder = 'topagnps_ohio_files/top2cche1d_outputs/' # Location to write the files
casename = 'ohio' # Name of the case for these files

img, geoMatrix, _, _, _, _ = t2c.read_esri_asc_file(filepath_annagnps_reach_ids)

dfagflow = t2c.read_agflow_reach_data(filepath_agflow)

df_nodes, df_channel, df_link, df_reach, df_csec, df_csprf, img_reach = t2c.convert_topagnps_output_to_cche1d_input(filepath_agflow, filepath_flovec, filepath_annagnps_reach_ids, filepath_netw, default_xsection, min_strahler, distance)

t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_nodes)
t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_channel)
t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_link)
t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_reach)
t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_csec)
t2c.write_cche1d_dat_file(f'{outputfolder}{casename}_min_strahler_{min_strahler}', df_csprf)
