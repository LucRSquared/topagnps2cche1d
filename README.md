# topagnps2cche1d (1.1.0)
Toolbox of Python functions to handle TopAGNPS outputs and reformat them as CCHE1D inputs

Please refer to the `top2cche1d_notebook.ipynb` for explanations and quick-start guide.


## Current features
Using the files produced by TopAGNPS:
  - `AnnAGNPS_Reach_Data_Section.csv`
  - `AnnAGNPS_Cell_Data_Section.csv`
  - Reach raster data: `AnnAGNPS_Reach_IDs.asc` (with accompanying projection file)

`topagnps2cche1d` will create a `Watershed` object containing the reaches and their connectivity in a NetworkX DiGraph.
- Reaches are represented with computational nodes with a structure compatible with the Fluvial Routing Analysis and Modeling Environment (FRAME) developed by Vieira (1997) 


## To-do:
- [ ] Producing CCHE1D Tables:
    - channel, reaches, links, nodes tables
    - boundary conditions tables (inflows from cells and "ghost" reaches)
  
- [ ] Implement reach resampling methods

- [ ] Implement processing reach geometryies from polygons GeoDataFrame (output of polygonization process)
  - Using reach skeletonization method

- [ ] Implement cross-sections methods


## Changelogs
- August 2023: 
  - Rewriting of the whole library in an object oriented programming approach. Removes the need for AgFlow and Strahler output data from TopAGNPS

## References:
Vieira, D.A., 1997, FRAME - Control Module Technical Manual, Tech. Rep. No. CCHE-TR-97-7, Center for Computational Hydroscience and Engineering, The University of Mississippi, University, Mississippi.
