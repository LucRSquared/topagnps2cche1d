import topagnps2cche1d.tools as t2c

# Location of all relevant files:
srcfolder = 'topagnps_ohio_files/'
filepath_annagnps_reach_ids = srcfolder + 'AnnAGNPS_Reach_IDs.asc'
filepath_agflow = srcfolder + 'AgFlow_Reach_Data.csv'
filepath_flovec = srcfolder + 'FLOVEC.asc'

outputfolder = 'topagnps_ohio_files/top2cche1d_outputs/' # Location to write the files
casename = 'ohio' # Name of the case for these files

img, geoMatrix, ncols, nrows, nodataval, dataset = t2c.read_esri_asc_file(filepath_annagnps_reach_ids)

dfagflow = t2c.read_agflow_reach_data(filepath_agflow)

df_nodes, df_channel, df_link, img_reach = t2c.convert_topagnps_output_to_cche1d_input(filepath_agflow, filepath_flovec, filepath_annagnps_reach_ids)

t2c.write_cche1d_dat_file(outputfolder+casename, df_nodes)
t2c.write_cche1d_dat_file(outputfolder+casename, df_channel)
t2c.write_cche1d_dat_file(outputfolder+casename, df_link)