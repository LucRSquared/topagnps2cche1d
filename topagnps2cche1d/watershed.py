import warnings
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
from topagnps2cche1d.tools import (
    custom_dfs_traversal_sorted_predecessors,
    find_extremities_pathgraph,
    get_pathgraph_binary,
    get_pathgraph_node_sequence,
    read_esri_asc_file,
    read_reach_data_section,
    read_cell_data_section,
    find_extremities_binary,
    get_intermediate_nodes_img,
)
from topagnps2cche1d import Reach, Node, Cell, CrossSection

import holoviews as hv
import hvplot.pandas


class Watershed:
    def __init__(self):
        self.reaches = {}
        self.cells = {}
        self.cross_sections = {}
        self.full_graph = None
        self.current_graph = None

        self.geomatrix = None

    def add_reach(self, reach):
        """
        Add a reach to the watershed
        """
        self.reaches[reach.id] = reach

    def add_cell(self, cell):
        """
        Add a cell to the watershed
        """
        self.cells[cell.id] = cell

    def add_cross_section(self, cross_section):
        """
        Add a cross section to the watershed
        """
        self.cross_sections[cross_section.id] = cross_section

    def import_topagnps_reaches_network(self, path_reach_data_section):
        """
        Imports information from "AnnAGNPS_Reach_Data_Section.csv" file produced by TopAGNPS
        and adds to Watershed
        """
        df = read_reach_data_section(path_reach_data_section)

        self._create_reaches_from_df(df)

        self.update_graph()

        self.assign_strahler_number_to_reaches()

    def import_topagnps_cells(self, path_cell_data_section):
        """
        Imports information from "AnnAGNPS_Cell_Data_Section.csv" file produced by TopAGNPS
        and adds to Watershed
        """

        df = read_cell_data_section(path_cell_data_section)
        df["Cell_Area"] = (
            df["Cell_Area"] * 1e4
        )  # TopAGNPS produces an area in ha, we want it in m²

        self._create_cells_from_df(df)

    def update_graph(self):
        """
        Replaces the connectivity dict by a NetworkX DiGraph
        Creates a fully connected graph from all the reaches that are in the watershed
        And then creates a subgraph -- current_graph from the reaches that are not ignored
        """
        reaches = self.reaches

        full_graph = nx.DiGraph()

        edges_full = []
        for reach_id, reach in reaches.items():
            # we don't add reaches that are not in the "Reach_ID" column
            if reach.receiving_reach_id not in reaches:
                continue
            edges_full.append((reach_id, reach.receiving_reach_id))

        full_graph.add_edges_from(edges_full)

        ignored_reaches = set()
        for reach_id, reach in reaches.items():
            if reach.ignore:
                ignored_reaches.add(reach_id)
                for ancestor_reach_id in nx.ancestors(full_graph, reach_id):
                    reaches[ancestor_reach_id].ignore_reach()
                    ignored_reaches.add(ancestor_reach_id)

        kept_reaches = set(full_graph.nodes) - ignored_reaches
        current_graph = full_graph.subgraph(kept_reaches)

        self.full_graph = full_graph
        self.current_graph = current_graph

    def update_watershed(self):
        """
        Updates:
            - Graph, Junctions, Node IDs, Node types, Renumbers everything in computational order
        """

        self.update_graph()

        if len(self.current_graph) == 0:
            raise Exception(
                "No reaches left to consider in Watershed connectivity graph"
            )

        
        self.renumber_all_nodes_and_reaches_in_CCHE1D_computational_order()
        self.set_node_id_to_compute_id()

        # Identify reaches with a single node and add two extra nodes, one upstream and one downstream
        self.add_us_and_ds_node_to_single_node_reaches()

        self.update_junctions_and_node_types()

        # # Identify reaches with a single node and add two extra nodes, one upstream and one downstream
        # self.add_us_and_ds_node_to_single_node_reaches()


        self.identify_inflow_sources()

        # self._print_reaches_node_ids()

        self.update_default_us_ds_default_values()
        self.renumber_all_nodes_and_reaches_in_CCHE1D_computational_order()
        self.set_node_id_to_compute_id()

    def keep_all_reaches(self):
        """
        This function sets all the reaches in the watershed to non ignored
        """
        reaches = self.reaches
        for reach in reaches.values():
            reach.ignore = False

    def ignore_reaches_with_strahler_leq(self, strahler_threshold=0):
        """
        Ignore reaches with Strahler Number less than a given threshold
        """
        self.keep_all_reaches()
        for reach in self.reaches.values():
            if reach.strahler_number <= strahler_threshold:
                reach.ignore_reach()

        self.update_graph()

    def assign_strahler_number_to_reaches(self, mode="fully_connected"):
        """
        Assign Strahler number to reaches in the system according to the connectivity dict.
        mode: 'fully_connected' (mode) -> Uses the fully connected network
              'current' -> Uses the current connectivity dict taking into account ignored reaches
        """

        reaches = self.reaches

        if mode == "fully_connected":
            graph = self.full_graph
        elif mode == "current":
            graph = self.current_graph
        else:
            warnings.warn(
                "Invalid mode to assign strahler number -> Using 'fully_connected' by default"
            )
            graph = self.full_graph

        # optimizing the order of reaches processed by using DFS algorithm
        queue = custom_dfs_traversal_sorted_predecessors(
            graph, start=None, visit_descending_order=True, postorder=True
        )

        # Process the queue
        while queue:
            current_id = queue.pop(0)  # take the first element
            current_reach = reaches[current_id]

            if graph.in_degree(current_id) == 0:
                # reach is at the most upstream end
                current_reach.strahler_number = 1
                continue

            # get list of upstream reaches
            upstream_reaches_strahler = [
                reaches[id].strahler_number for id in graph.predecessors(current_id)
            ]

            if None in upstream_reaches_strahler:
                # undetermined case, keep current_reach in queue
                queue.append(current_id)
                continue
            else:
                max_strahler = max(upstream_reaches_strahler)
                count_max_strahler = upstream_reaches_strahler.count(max_strahler)

                if count_max_strahler >= 2:
                    current_reach.strahler_number = max_strahler + 1
                else:
                    current_reach.strahler_number = max_strahler

    def get_highest_strahler_number(self):
        """
        Returns the highest Strahler number in the watershed
        """

        reaches = self.reaches
        highest_strahler = 0
        for reach in reaches.values():
            if reach.strahler_number > highest_strahler:
                highest_strahler = reach.strahler_number
        return highest_strahler


    def read_reaches_geometry_from_polygons_gdf(self, gdf, column_id_name="dn"):
        """
        Takes a GeoDataFrame containing POLYGONS describing reaches (most likely from gdal_polygonize of AnnAGNPS_Reach_IDs.asc raster)
        and identifies the skeleton and adds nodes along the paths of every reach.
        The reaches returned are not necessarily in the correct US/DS order
        """
        pass

    def read_reaches_geometry_from_topagnps_asc_file(self, path_to_asc_file):
        """
        - This function takes as input the "AnnAGNPS_Reach_IDs.asc" file and adds to the watershed
          the gdal geomatrix
        - For every raster being a part of a reach it adds a node to the corresponding reach
        """

        img_reach_asc, geomatrix, _, _, nodataval_reach_asc, _ = read_esri_asc_file(
            path_to_asc_file
        )
        # raster_size = abs(geomatrix[1])

        self.geomatrix = geomatrix

        reaches = self.reaches

        nd_counter = 0

        # Get list of reach ids (excluding nodataval_reach_asc = 0 typically)
        mask_no_data = img_reach_asc == nodataval_reach_asc
        list_of_reaches_in_raster = set(
            np.unique(img_reach_asc[~mask_no_data]).astype(int)
        )

        for reach_id in tqdm(list_of_reaches_in_raster, desc="Reading Reaches"):
            reach_img = np.where(img_reach_asc == reach_id, 1, 0)
            # at this point we don't know if the start and end are the upstream and downstream

            connected_pixels_graph = get_pathgraph_binary(
                reach_img, output_index_starts_one=False
            )
            extremities = find_extremities_pathgraph(connected_pixels_graph)

            if len(extremities) == 1:
                # Case where the reach is only one pixel
                rowcol = extremities[0]
                nd_counter += 1
                reaches[reach_id].add_node(
                    Node(id=nd_counter, row=rowcol[0], col=rowcol[1])
                )
                reaches[reach_id].us_nd_id = nd_counter
                reaches[reach_id].ds_nd_id = nd_counter

                continue
            elif len(extremities) == 2:
                startrowcol, endrowcol = extremities
            else:
                raise Exception(
                    f"Invalid reach {reach_id} in raster, more than 2 extremities"
                )

            reach_skel_rowcols = get_pathgraph_node_sequence(
                connected_pixels_graph, source=startrowcol, end=endrowcol
            )
            # reach_skel_rowcols = get_intermediate_nodes_img(
            #     startrowcol, endrowcol, reach_img
            # )

            for k, pixel_rowcol in enumerate(reach_skel_rowcols, 1):
                nd_counter += 1
                if k == 1:
                    # start node
                    reaches[reach_id].add_node(
                        Node(
                            id=nd_counter,
                            usid=None,
                            dsid=nd_counter + 1,
                            row=pixel_rowcol[0],
                            col=pixel_rowcol[1],
                        )
                    )
                    reaches[reach_id].us_nd_id = nd_counter
                elif k == len(reach_skel_rowcols):
                    # end node
                    reaches[reach_id].add_node(
                        Node(
                            id=nd_counter,
                            usid=nd_counter - 1,
                            dsid=None,
                            row=pixel_rowcol[0],
                            col=pixel_rowcol[1],
                        )
                    )
                    reaches[reach_id].ds_nd_id = nd_counter
                else:  # middle nodes
                    reaches[reach_id].add_node(
                        Node(
                            id=nd_counter,
                            usid=nd_counter - 1,
                            dsid=nd_counter + 1,
                            row=pixel_rowcol[0],
                            col=pixel_rowcol[1],
                        )
                    )

        # Set the reaches that are not in the raster to be removed from the reaches and graph
        all_reaches_ids = set(reaches.keys())
        reaches_to_remove = all_reaches_ids - list_of_reaches_in_raster
        for ri in reaches_to_remove:
            del reaches[ri]

        self.compute_XY_coordinates_of_all_nodes(oneindexed=False)
        self.update_graph()
        self.determine_reaches_us_ds_direction()
        # self.add_us_and_ds_node_to_single_node_reaches()

    def add_us_and_ds_node_to_single_node_reaches(self):
        """
        Looks for single node reaches and adds an upstream and downstream node to them
        """

        reaches = self.reaches

        nd_counter = self._get_highest_node_id() + 1

        # DEALING WITH CASES WHERE REACHES ONLY HAVE ONE NODE

        for reach_id in reaches:
            reach_before = reaches[reach_id]

            if reach_before.ignore:
                continue

            reach_middle_id = reach_before.receiving_reach_id
    
            if (reach_middle_id is None) or (reach_middle_id not in reaches):
                # reach before outlet case
                continue
            else:
                reach_middle = reaches[reach_middle_id]
                
                if reach_middle.ignore:
                    continue

                reach_after_id = reach_middle.receiving_reach_id

            if (reach_after_id is None) or (reach_after_id not in reaches):
                # reach before outlet case
                continue
            else:
                reach_after = reaches[reach_after_id]

                if reach_after.ignore:
                    continue


            # If the middle has only one node we need to artificially add two more in between the others
            if len(reach_middle.nodes)==1:
                node_reach_before_ds = reach_before.nodes[reach_before.ds_nd_id]
                node_middle = reach_middle.nodes[reach_middle.us_nd_id]
                node_reach_after_us = reach_after.nodes[reach_after.us_nd_id]

                # Create two new nodes that fit inbetween the middle node and the neighbors us and ds
                nd_counter += 1
                reach_middle.add_node(
                    Node(
                        id=nd_counter,
                        usid=None,
                        dsid=node_middle.id,
                        x=(node_reach_before_ds.x+node_middle.x)/2,
                        y=(node_reach_before_ds.y+node_middle.y)/2
                    )
                )
                reach_middle.us_nd_id = nd_counter

                node_middle.usid = nd_counter

                nd_counter += 1
                reach_middle.add_node(
                    Node(
                        id=nd_counter,
                        usid=node_middle.id,
                        dsid=None,
                        x=(node_reach_after_us.x+node_middle.x)/2,
                        y=(node_reach_after_us.y+node_middle.y)/2
                    )
                )
                reach_middle.ds_nd_id = nd_counter

                node_middle.dsid = nd_counter

            else:
                continue

    def compute_XY_coordinates_of_all_nodes(self, oneindexed=False):
        """
        Compute the XY coordinates of all the nodes in the Watershed
        - oneindexed=False assumes that the row col coordinates are provided in the 0-starting format
        """

        reaches = self.reaches
        geomatrix = self.geomatrix

        for reach in reaches.values():
            reach.compute_XY_coordinates_of_reach_nodes(
                geomatrix, oneindexed=oneindexed
            )

    def determine_reaches_us_ds_direction(self):
        """
        This function looks at the nodes in each reach of the watershed
        and based on the reach connectivity determines which end of each reach
        is downstream or upstream and updates the reach (us/ds)_nd_id
        """

        current_graph = self.current_graph
        reaches = self.reaches

        # We know that the order of these is from upstream to downstream,
        # the last one being the reach just before the outlet
        reach_order = custom_dfs_traversal_sorted_predecessors(
            current_graph, start=None, visit_descending_order=True, postorder=True
        )

        for reach_id in reach_order:
            receiving_reach_id = reaches[reach_id].receiving_reach_id
            if (receiving_reach_id is None) or (receiving_reach_id not in reaches):
                # reach before outlet case
                continue

            receiving_reach = reaches[receiving_reach_id]

            RR_pot_usnd_id = receiving_reach.us_nd_id
            RR_pot_dsnd_id = receiving_reach.ds_nd_id

            RR_pot_usnd = receiving_reach.nodes[RR_pot_usnd_id]
            RR_pot_dsnd = receiving_reach.nodes[RR_pot_dsnd_id]

            current_reach = reaches[reach_id]

            CR_pot_usnd_id = current_reach.us_nd_id
            CR_pot_dsnd_id = current_reach.ds_nd_id

            CR_pot_usnd = current_reach.nodes[CR_pot_usnd_id]
            CR_pot_dsnd = current_reach.nodes[CR_pot_dsnd_id]

            dist_CR_ds_RR_us = CR_pot_dsnd.distance_from(RR_pot_usnd)
            dist_CR_us_RR_us = CR_pot_usnd.distance_from(RR_pot_usnd)
            dist_CR_ds_RR_ds = CR_pot_dsnd.distance_from(RR_pot_dsnd)
            dist_CR_us_RR_ds = CR_pot_usnd.distance_from(RR_pot_dsnd)

            distances = [
                dist_CR_us_RR_us,
                dist_CR_ds_RR_us,
                dist_CR_us_RR_ds,
                dist_CR_ds_RR_ds,
            ]

            min_dist = min(distances)

            if min_dist == dist_CR_ds_RR_us:
                # do nothing, both reaches are correctly ordered
                continue
            elif min_dist == dist_CR_us_RR_us:
                current_reach.flip_reach_us_ds_order()
                # don't flip receiving_reach
            elif min_dist == dist_CR_ds_RR_ds:
                receiving_reach.flip_reach_us_ds_order()
            elif min_dist == dist_CR_us_RR_ds:
                current_reach.flip_reach_us_ds_order()
                receiving_reach.flip_reach_us_ds_order()

    def resample_reaches(self, id_list="all", **kwargs):
        """
        Resample specific reach, list of reaches, or all reaches either at a constant spacing or a given number of points.
        ### positional argument:
        - id_list: id of reach(es) to resample OR 'all' to resample all reaches

        ### key-value arguments:
        - step : define a step length (in the units of x and y) to resample points along the cord length
        OR
        - numpoints: define an arbitrary number of points to resample along the cord length
        - min_numpoints: define a minimum number of points to use (default: 3)
        WARNING: If both values are specified, the method yielding the greater of resampling nodes will be chosen
        It is advised to use only one of the keywords arguments.
        """
        if isinstance(id_list, int):
            id_list = [id_list]
        elif id_list == "all":
            id_list = self.reaches
        elif not (isinstance(id_list, list) or isinstance(id_list, tuple)):
            raise Exception("Invalid argument for list of reaches to resample")

        for reach_id in id_list:
            self.reaches[reach_id].resample_reach(
                **kwargs, nodes_new_id_start=self._get_highest_node_id() + 1
            )

        # Update node ids, junctions, etc.
        self.update_watershed()

    def identify_inflow_sources(self):
        """
        There are two types of inflow sources:
            1. Cells all throughout the watershed pouring into either source nodes
               or the downstream node of their adjacent reach
            2. Inflow from a "ghost" reach that was removed
        """
        self._find_cells_inflow_nodes()
        self._find_removed_reaches_inflow_nodes()

    def update_junctions_and_node_types(self):
        """
        For the current connectivity graph:
            - updates junctions
            - set nodes type
        """
        self._create_junctions_between_reaches()
        self._set_outlet_node_type()
        self._set_inflow_nodes_type()
        self._set_default_node_type()

    def renumber_all_nodes_and_reaches_in_CCHE1D_computational_order(self):
        """
        This function renumbers all the nodes and reaches according to their CCHE1D computational order
        """
        current_graph = self.current_graph
        reaches = self.reaches

        reach_dfs_postorder = custom_dfs_traversal_sorted_predecessors(
            current_graph, visit_descending_order=True, postorder=True
        )

        computeid = 0
        for cche1d_id, reach_id in enumerate(reach_dfs_postorder, 1):
            reach = reaches[reach_id]
            nodes = reach.nodes
            reach.cche1d_id = cche1d_id
            us_node_id = reach.us_nd_id
            ds_node_id = reach.ds_nd_id
            current_node_id = us_node_id

            # print(f"Renumbering CCHE1D reach: {cche1d_id}, TopAGNPS reach: {reach_id}")

            while True:
                computeid += 1
                current_node = nodes[current_node_id]
                current_node.computeid = computeid

                if current_node_id == ds_node_id:
                    break
                else:
                    current_node_id = current_node.dsid

        # I should also renumber us_nd_id and ds_nd_id

    def set_node_id_to_compute_id(self):
        """
        Optional, changes node absolute id to its computational id
        """
        reaches = self.reaches

        old_new_dict = {}

        for reach in reaches.values():
            nodes = reach.nodes
            for node in nodes.values():
                old_new_dict[node.id] = node.computeid

        # Apply dictionary
        for reach in reaches.values():
            reach.change_node_ids_dict(old_new_dict)
            nodes = reach.nodes
            for node in nodes.values():
                node.change_node_ids_dict(old_new_dict)

    def assign_cross_section_to_node_byid(self, node_id, **kwargs):
        """
        Define a cross section in the local transverse plane coordinates along the channel. The coordinates are given in the left to right
        format (when looking downstream)
        Arguments:
            node_id : id of the node we want to assign the cross section to

        Key-Value Arguments:
        * node_id_type : node id type
            - 'absolute' corresponds to the "id" attribute (default)
            - 'computeid' corresponds to the "computeid" attribute
        * cs_id: cross section unique id
            If not provided it will increment the highest cross section id by one
        * cs_type:
            - 'default' uses a trapezoidal with flood plain cross section made of 8 points with arbitrary values
            - 'trapezoidal_with_flood_plain' which accepts the parameters defined below
                * 'bottom_elevation'          : elevation of bottom of channel
                * 'flood_plain_width'         : width of the flood plain on each side
                * 'main_channel_bottom_width' : width of the bottom of the main channel
                * 'main_channel_depth'        : depth of main channel measured from bottom to elevation of flood plain
                * 'main_channel_bank_slope'   : slope of main channel banks
                * 'flood_plain_bank_slope'    : slope of flood plain banks
                * 'flood_plain_bank_height    : height of flood plain bank
            - 'user_defined' : the user provides the coordinates ws, zs, and lfp_idx, lbt_idx, rbt_idx, rfp_idx
                if the indexes are not provided then they are defaulted to end and beginning indexes
        * ws: cross section abscissas arrays from left to right
        * zs: cross section ordinates arrays
        * lfp_idx: 'Left Flood Plain' edge Index of cross section marking the beginning of the main channel
        * lbt_idx: 'Left Bank Toe' Index of cross section marking the toe of the left bank
        * rbt_idx: 'Right Bank Toe'
        * rfp_idx: 'Right Flood Plain' Index marking the end of the main channel on the right bank
        /!\ these indexes start at 1 (ONE) not 0 (ZERO)
        * mannings_roughness : Manning's Roughness coefficient
            Either a float representing the roughness of the entire cross section
            or an array of length len(zs) where n_rgh[i] is the roughness of point (ws[i], zs[i])

        e.g. watershed.assign_cross_section_to_node_byid(12)
        """

        if "node_id_type" in kwargs:
            node_id_type = kwargs["node_id_type"]
        else:
            node_id_type = "absolute"

        if "cs_id" in kwargs:
            cs_id = kwargs["cs_id"]
        else:
            cs_id = self._get_highest_cross_section_id() + 1

        if "cs_type" in kwargs:
            cs_type = kwargs["cs_type"]
        else:
            cs_type = "trapezoidal_with_flood_plain"

        cross_section = CrossSection(id=cs_id, type=cs_type, **kwargs)

        node = self._get_node_byid(node_id, id_type=node_id_type)

        # Add cross section to Watershed collection of cross sections
        self.add_cross_section(cross_section)
        # Assign cross section id to node
        node.csid = cs_id

    def assign_cross_section_to_all_points(self, **kwargs):
        """
        This function accepts the same key-value arguments as the function assign_cross_section_to_node_byid
        except for node_id, node_id_type, and cs_id. The function will assign the defined cross section to all points in the current watershed
        """

        if "cs_type" in kwargs:
            cs_type = kwargs["cs_type"]
        else:
            cs_type = "trapezoidal_with_flood_plain"

        reaches = self.reaches
        current_graph = self.current_graph

        cs_id = 0
        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            for node in reach.nodes.values():
                cs_id += 1
                cross_section = CrossSection(id=cs_id, type=cs_type, **kwargs)
                node.csid = cs_id
                self.add_cross_section(cross_section)

    def reset_cross_sections_elevation(self, elevation=0):
        """
        Resets the thalweg elevation of all cross sections in the watershed.
        Default to 0 but can be specified to a constant value
        """

        for cross_section in self.cross_sections.values():
            cross_section.set_thalweg_elevation(elevation)

    def adjust_cross_sections_elevation_with_slope(self, outlet_elevation=0):
        """
        Traverse the watershed network upstream and adjust the elevation of the thalweg
        of each cross section
        ### Parameter:
            - outlet_elevation: float, optional reference elevation of the outlet
        """
        reaches = self.reaches
        cross_sections = self.cross_sections
        compute_id_reach_id = {reach.cche1d_id: reach.id for reach in reaches.values() if reach.cche1d_id is not None}
        max_compute_id = max(compute_id_reach_id.keys())
        outlet_reach_id = compute_id_reach_id[max_compute_id]

        # Initialize loop
        outlet_reach = reaches[outlet_reach_id]  # Outlet ID
        outlet_node = outlet_reach.nodes[outlet_reach.ds_nd_id]  # Outlet node

        # Get list of reaches from downstream to upstream in a Breadth-First-Search order
        # Here we use .reverse() on the graph because conceptually the bfs algorithm works in verse
        # from the upstream/downstream logic. Going "deeper" means going down the paths but it is
        reaches_to_visit_going_upstream = list(
            nx.bfs_tree(self.current_graph.reverse(), outlet_reach_id)
        )

        for reach_id in reaches_to_visit_going_upstream:
            # Get reach
            current_reach = reaches[reach_id]
            # Identify the ID of its DS and US nodes
            ds_nd_id = current_reach.ds_nd_id
            us_nd_id = current_reach.us_nd_id
            # Get list of nodes
            nodes = current_reach.nodes

            # Get previous node
            if ds_nd_id == outlet_node.id:
                previous_node = outlet_node
                elevation = outlet_elevation  # Initial elevation
            else:
                ds_reach = reaches[current_reach.receiving_reach_id]
                previous_node = ds_reach.nodes[ds_reach.us_nd_id]
                # Restart elevation at the elevation of the previous node
                elevation = cross_sections[previous_node.csid].get_thalweg_elevation()

            # Traverse nodes starting ds
            current_node_id = ds_nd_id
            while True:
                # Get current node
                current_node = nodes[current_node_id]

                # Compute distance from previous node
                dist_prev_node = current_node.distance_from(previous_node)
                # Compute elevation of current node based on slope
                elevation += dist_prev_node * current_reach.slope

                # Get node cross section
                cross_section = cross_sections[current_node.csid]
                # Shift current node cross section according to absolute elevation
                cross_section.set_thalweg_elevation(elevation)

                if current_node_id == us_nd_id:
                    break
                else:
                    current_node_id = current_node.usid
                    previous_node = current_node

    def create_cche1d_nodes_df(self):
        """
        Generate a DataFrame containing all the nodes in CCHE1D format
        """

        reaches = self.reaches
        current_graph = self.current_graph

        # Defining columns
        nd_ID, nd_FRMNO = [], []
        nd_TYPE = []
        nd_XC, nd_YC = [], []
        nd_DSID, nd_USID, nd_US2ID = [], [], []
        nd_CSID = []
        nd_RSID = []  # Legacy Raster cell record ID set to -1
        nd_STID = []  # Hydraulic structure record ID set to 1 by default

        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            for node in reach.nodes.values():
                nd_ID.append(node.id)
                nd_FRMNO.append(node.computeid)
                nd_TYPE.append(node.type)
                nd_XC.append(node.x)
                nd_YC.append(node.y)
                nd_DSID.append(node.dsid)
                nd_USID.append(node.usid)
                nd_US2ID.append(node.us2id)
                nd_CSID.append(node.csid)
                nd_RSID.append(-1)  # set a default value
                nd_STID.append(1)  # set de default value

        df = pd.DataFrame(
            {
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
        )

        df["ND_ID"] = df["ND_ID"].astype(int)
        df["ND_FRMNO"] = df["ND_FRMNO"].astype(int)
        df["ND_TYPE"] = df["ND_TYPE"].astype("category")
        df["ND_XC"] = df["ND_XC"].astype(float)
        df["ND_YC"] = df["ND_YC"].astype(float)
        df["ND_DSID"] = df["ND_DSID"].astype(int)
        df["ND_USID"] = df["ND_USID"].astype(int)
        df["ND_US2ID"] = df["ND_US2ID"].astype(int)
        df["ND_CSID"] = df["ND_CSID"].astype(int)
        df["ND_RSID"] = df["ND_RSID"].astype(int)
        df["ND_STID"] = df["ND_STID"].astype(int)

        df = df.sort_values(by="ND_ID")

        df.reset_index(inplace=True, drop=True)

        return df

    def create_cche1d_channels_df(self):
        """
        Generate a DataFrame containing all the TopAGNPS reaches (called channels for CCHE1D) in CCHE1D format
        """

        reaches = self.reaches
        current_graph = self.current_graph

        # Define columns
        ch_ID = []
        ch_NDUSID, ch_NDDSID = [], []
        ch_LENGTH = []

        for reach_id, reach in reaches.items():
            if reach_id in current_graph:
                ch_ID.append(reach.cche1d_id)
                ch_NDUSID.append(reach.us_nd_id)
                ch_NDDSID.append(reach.ds_nd_id)
                ch_LENGTH.append(reach.length())

        df = pd.DataFrame(
            {
                "CH_ID": ch_ID,
                "CH_NDUSID": ch_NDUSID,
                "CH_NDDSID": ch_NDDSID,
                # "CH_LENGTH": ch_LENGTH,  # NOTE: See if this causes problems when importing in GUI
            }
        )

        df = df.sort_values(by="CH_ID")

        df.reset_index(inplace=True, drop=True)

        return df

    def create_cche1d_links_and_reaches_df(self):
        """
        Generate a DataFrame containing all the links and reaches in CCHE1D format
        /!\ A CCHE1D Reach is not the same as a TopAGNPS reach. A CCHE1D reach is simply
        a section between two nodes
        """

        reaches = self.reaches
        current_graph = self.current_graph

        # Define columns
        lk_ID = []
        lk_NDUSID, lk_NDDSID = [], []
        lk_RCUSID, lk_RCDSID = [], []
        lk_TYPE = []  # Link type, 1 = channel (default), 2 = hydraulic strucutre
        lk_LENGTH = []

        rc_ID = []
        rc_NDUSID, rc_NDDSID = [], []
        rc_ORDER = []  # Strahler order
        rc_SLOPE = []
        rc_LENGTH = []

        # rc_id = 0
        rc_id = 1
        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue

            nodes = reach.nodes
            # Create cche1d reaches
            current_node_id = reach.us_nd_id
            current_node = nodes[current_node_id]
            # us_rc_id = max(
            #     1, rc_id
            # )  # this is so that if rc_id = 0 then we know that the US reach ID in the CCHE1D sense of the term is 1
            us_rc_id = rc_id
            while current_node_id != reach.ds_nd_id:
                # rc_id += 1
                rc_ID.append(rc_id)
                rc_NDUSID.append(current_node.id)
                rc_NDDSID.append(current_node.dsid)
                rc_ORDER.append(reach.strahler_number)
                rc_SLOPE.append(reach.slope)

                ds_node = nodes[current_node.dsid]
                rc_LENGTH.append(current_node.distance_from(ds_node))
                current_node = ds_node
                current_node_id = ds_node.id
                rc_id += 1  # update for the next reach (if there is one)

            ds_rc_id = (
                rc_id - 1
            )  # rc_id-1 is the downstream CCHE1D reach id for that topagnps reach

            lk_ID.append(reach.cche1d_id)
            lk_NDUSID.append(reach.us_nd_id)
            lk_NDDSID.append(reach.ds_nd_id)
            lk_RCUSID.append(us_rc_id)
            lk_RCDSID.append(ds_rc_id)
            lk_TYPE.append(
                1
            )  # by default we don't consider hydraulic structures so the link is a channel
            lk_LENGTH.append(
                reach.length()
            )  # NOTE : See if it causes issues when importing in GUI

        df_lk = pd.DataFrame(
            {
                "LK_ID": lk_ID,
                "LK_CMPSEQ": lk_ID,
                "LK_NDUSID": lk_NDUSID,
                "LK_NDDSID": lk_NDDSID,
                "LK_RCUSID": lk_RCUSID,
                "LK_RCDSID": lk_RCDSID,
                "LK_TYPE": lk_TYPE
                # "LK_LENGTH": lk_LENGTH, # NOTE : didn't use to include this
            }
        )

        df_lk = df_lk.sort_values(by="LK_ID")
        df_lk.reset_index(inplace=True, drop=True)

        df_rc = pd.DataFrame(
            {
                "RC_ID": rc_ID,
                "RC_NDUSID": rc_NDUSID,
                "RC_NDDSID": rc_NDDSID,
                "RC_ORDER": rc_ORDER,
                # "RC_SLOPE": rc_SLOPE,  # NOTE : didn't use to include this, we'll see if it causes issues when importing in GUI
                "RC_LENGTH": rc_LENGTH,
            }
        )

        df_rc.reset_index(inplace=True, drop=True)

        return df_lk, df_rc

    def create_cche1d_csec_csprf_df(self):
        """
        Create the two DataFrames that describe the cross sections
        """

        reaches = self.reaches
        cross_sections = self.cross_sections
        current_graph = self.current_graph

        # Define columns: csec
        cs_ID = []
        cs_NPTS = []
        cs_LOB, cs_LBT, cs_ROB, cs_RBT = [], [], [], []
        cs_SVTYPE = []  # cs type, we provide wz so it will always be 'WZ'
        cs_TYPE = (
            []
        )  # Can be 'MC', 'MCLF', 'MCRF', 'MCLFRF' depending if floodplains are provided. for now we'll just put 'MC'
        cs_ORIGIN = []  # Constant will be 'USERSPEC'
        cs_STATION = []  # header needs to be there but will be empty

        # Define columns: csprf
        cp_ID = []
        cp_CSID = []
        cp_POSIDX = []
        cp_W, cp_Z, cp_RGH = [], [], []
        cp_BLOCK = []  # not sure what this is but it is set constant as 1

        cp_id = 0  # global counter for cross section points
        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            nodes = reach.nodes
            for node in nodes.values():
                csid = node.csid
                xsection = cross_sections[csid]
                ws, zs, n_rgh = xsection.ws, xsection.zs, xsection.n_rgh
                lfp, lbt, rbt, rfp = (
                    xsection.lfp_idx,
                    xsection.lbt_idx,
                    xsection.rbt_idx,
                    xsection.rfp_idx,
                )

                numpts = len(ws)

                cp_id += 1
                cp_ID.extend(range(cp_id, cp_id + numpts))
                cp_CSID.extend([csid for _ in range(numpts)])
                cp_POSIDX.extend(range(1, 1 + numpts))
                cp_W.extend(ws)
                cp_Z.extend(zs)
                cp_RGH.extend(n_rgh)
                cp_BLOCK.extend([1 for _ in range(numpts)])
                cp_id = cp_id + numpts - 1  # is also equal to the cp_ID[-1]

                cs_ID.append(csid)
                cs_NPTS.append(numpts)
                cs_LOB.append(lfp)
                cs_LBT.append(lbt)
                cs_RBT.append(rbt)
                cs_ROB.append(rfp)
                cs_TYPE.append("MC")
                cs_SVTYPE.append("WZ")
                cs_ORIGIN.append("USERSPEC")
                cs_STATION.append("")

        df_csec = pd.DataFrame(
            {
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
        )

        df_csec = df_csec.sort_values(by="CS_ID")
        df_csec.reset_index(inplace=True, drop=True)

        df_csprf = pd.DataFrame(
            {
                "CP_ID": cp_ID,
                "CP_CSID": cp_CSID,
                "CP_POSIDX": cp_POSIDX,
                "CP_W": cp_W,
                "CP_Z": cp_Z,
                "CP_RGH": cp_RGH,
                "CP_BLOCK": cp_BLOCK,
            }
        )

        df_csprf.reset_index(inplace=True, drop=True)

        return df_csec, df_csprf

    def create_cche1d_twcells_df(self):
        """
        Create a DataFrame of the cells pouring into the CCHE1D reach network
        """
        cells = self.cells
        reaches = self.reaches
        current_graph = self.current_graph

        tw_ID = []  # cell absolute id
        tw_NUMB = []  # cell TopAGNPS Cell_ID
        tw_NDDSID = []  # by default cells are defined to pour into the downstream node
        tw_NDUSID = []  # we also include the current reach's upstream node id
        tw_DRAREA = []  # we include the cell area

        tw_id = 0
        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            nodes = reach.nodes

            for node in nodes.values():
                for cell_source_id in node.inflow_cell_sources:
                    tw_id += 1

                    cell_source = cells[cell_source_id]
                    tw_ID.append(tw_id)
                    tw_NUMB.append(cell_source_id)
                    tw_NDDSID.append(node.id)
                    tw_NDUSID.append(
                        reach.us_nd_id
                    )  # For convenience we include the reach upstream node
                    tw_DRAREA.append(cell_source.area)

        df = pd.DataFrame(
            {
                "TW_ID": tw_ID,
                "TW_NUMB": tw_NUMB,
                "TW_NDDSID": tw_NDDSID,
                "TW_NDUSID": tw_NDUSID,
                "TW_DRAREA": tw_DRAREA,
            }
        )

        return df

    def get_list_of_inflow_reaches_df(self):
        """
        Create a DataFrame with a list of nodes that have inflow reaches that were removed
        from an AnnAGNPS simulation
        """

        reaches = self.reaches
        current_graph = self.current_graph

        # Defining columns
        inflow_reach_ID = []
        receiving_node_ID = []

        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            nodes = reach.nodes

            for node in nodes.values():
                for inflow_reach in node.inflow_reach_sources:
                    inflow_reach_ID.append(inflow_reach)
                    receiving_node_ID.append(node.id)

        df = pd.DataFrame(
            {
                "inflow_topagnps_reach_id": inflow_reach_ID,
                "receiving_node_id": receiving_node_ID,
            }
        )
        df = df.sort_values(by="inflow_topagnps_reach_id")

        return df

    def write_cche1d_dat_files(
        self, casename="topagnps", output_folder=None, float_format="%.4f", sep="\t"
    ):
        """
        Generates all the CCHE1D dat files in the specified folder.
        This function writes the CCHE1D files :
            * {casename}_channel.dat
            * {casename}_csec.dat
            * {casename}_csprf.dat
            * {casename}_link.dat
            * {casename}_nodes.dat
            * {casename}_reach.dat
            * {casename}_tw_cell.dat
        As well as a list_of_inflow_reaches.csv file to know which reach outputs need to be generated in AnnAGNPS if
        some of them were removed from the network to provide correct inflow BCs to CCHE1D
        """

        self.assign_strahler_number_to_reaches(mode="current")

        def _write_df(df, filename, float_format=float_format, sep=sep):
            # Little helper function
            filename.write_text(f"{df.shape[0]}\n")
            df.to_csv(
                filename, mode="a", index=False, float_format=float_format, sep=sep
            )

        if output_folder is None:
            output_folder = Path().cwd()
        else:
            output_folder = Path(output_folder)

        output_folder.mkdir(exist_ok=True)

        file = Path(output_folder) / casename

        # Generating dfs
        df_nodes = self.create_cche1d_nodes_df()
        df_ch = self.create_cche1d_channels_df()
        df_lk, df_rc = self.create_cche1d_links_and_reaches_df()
        df_csec, df_csprf = self.create_cche1d_csec_csprf_df()
        df_tw = self.create_cche1d_twcells_df()

        df_inflow_reaches = self.get_list_of_inflow_reaches_df()

        # Writing files
        file_nodes = file.with_name(f"{casename}_nodes.dat")
        _write_df(df_nodes, file_nodes)

        file_channels = file.with_name(f"{casename}_channel.dat")
        _write_df(df_ch, file_channels)

        file_links = file.with_name(f"{casename}_link.dat")
        _write_df(df_lk, file_links)

        file_reaches = file.with_name(f"{casename}_reach.dat")
        _write_df(df_rc, file_reaches)

        file_csec = file.with_name(f"{casename}_csec.dat")
        _write_df(df_csec, file_csec)

        file_csprf = file.with_name(f"{casename}_csprf.dat")
        _write_df(df_csprf, file_csprf)

        file_tw = file.with_name(f"{casename}_tw_cell.dat")
        _write_df(df_tw, file_tw)

        if df_inflow_reaches.size != 0:
            file_inflow_reaches = file.with_name(f"{casename}_inflow_reaches.csv")
            df_inflow_reaches.to_csv(file_inflow_reaches, index=False)

        self.assign_strahler_number_to_reaches(mode="fully_connected")

    def plot(self, **kwargs):
        """
        Plot network
        """
        reaches_plots = []

        if "title" in kwargs:
            title = kwargs["title"]
        else:
            title = ""

        if "aspect" in kwargs:
            aspect = kwargs["aspect"]
        else:
            aspect = "equal"

        if "frame_width" in kwargs:
            frame_width = kwargs["frame_width"]
        else:
            frame_width = 1000

        if "frame_height" in kwargs:
            frame_height = kwargs["frame_height"]
        else:
            frame_height = 1000

        if "by" in kwargs:
            by = kwargs["by"]
        else:
            by = "Reach_ID"

        if "line_width" in kwargs:
            line_width = kwargs["line_width"]
        else:
            line_width = 0

        current_graph = self.current_graph

        for reach_id, reach in self.reaches.items():
            if reach_id in current_graph:
                reaches_plots.append(reach.get_nodes_as_df())

        dfs = pd.concat(reaches_plots, ignore_index=True)

        watershed_plot = (
            dfs.hvplot.line(
                x="X",
                y="Y",
                by=by,
                hover_cols=[
                    "TYPE",
                    "US2ID",
                    "USID",
                    "ID",
                    "COMPUTEID",
                    "DSID",
                    "CCHE1D_ID",
                    "Reach_ID",
                ],
                line_width=line_width,
            )
            * dfs.hvplot(
                x="X",
                y="Y",
                by=by,
                kind="scatter",
                alpha=0.5,
                hover_cols=[
                    "TYPE",
                    "US2ID",
                    "USID",
                    "ID",
                    "COMPUTEID",
                    "DSID",
                    "CCHE1D_ID",
                    "Reach_ID",
                ],
            )
        ).opts(
            frame_width=frame_width,
            frame_height=frame_height,
            aspect=aspect,
            title=title,
        )

        return watershed_plot

    def update_default_us_ds_default_values(self):
        """
        After junctions the rest of the nodes need to be set to a default value for CCHE1D
        """
        reaches = self.reaches

        for reach in reaches.values():
            nodes = reach.nodes
            for node in nodes.values():
                if node.type == 9:  # if it's the outlet node then it's its own ds node
                    node.dsid = node.id

                if node.us2id is None:
                    node.us2id = -1

                if node.usid is None:
                    node.usid = -1

                if node.dsid is None:
                    node.dsid = -1

    def _create_junctions_between_reaches(self):
        """
        Every reach that has exactly two upstream reaches needs to have its most upstream node added as the most downstream node
        (= 'end of link' node with type = 3). The DSID, USID, US2ID need to be updated accordingly.
        In two inflows junction, the second inflow (which is the last reach upstream of the downstream reach when numbered in a depth
        first search method with a right hand rule (looking upstream)). The second inflow is thus also the reach with the smallest reach_id
        value according to TopAGNPS numbering system.

        If a reach has only one predecessor it's a trivial connection and the DSID, USID, and TYPE can be easily handled.
        """

        latest_nd_id = self._get_highest_node_id()
        reaches = self.reaches
        current_graph = self.current_graph

        # Remove type 3 nodes if they exist
        for reach in reaches.values():
            nodes_to_delete = []
            nodes = reach.nodes
            for node_id, node in nodes.items():
                if (node.type == 3) and len(
                    list(current_graph.predecessors(reach.receiving_reach_id))
                ) >= 2:
                    # We only delete the node if it's the end of a reach of a proper junction
                    # i.e. if the reach downstream has at least two inflow reaches. Otherwise we keep it
                    # for the case of simple junctions
                    usid = node.usid
                    reach.ds_nd_id = usid
                    # the new ds_nd_id is the node that was immediately before

                    nodes[usid].dsid = None
                    # the node immediately before now has no dsid

                    nodes_to_delete.append(node_id)

                if node.type == 2:
                    node.type = None
            for node_id in nodes_to_delete:
                del nodes[node_id]  # this is not working somehow

        for reach_id in current_graph.nodes:
            # getting current reach and its most upstream node
            reach = reaches[reach_id]
            ds_reach_us_junc_node = reach.nodes[reach.us_nd_id]

            # getting the list of upstream reaches in topagnps ascending order (CCHE1D descending)
            upstream_reaches_id = sorted(list(current_graph.predecessors(reach_id)))

            if len(upstream_reaches_id) == 0:
                # No upstream reaches
                continue
            elif len(upstream_reaches_id) == 1:
                # Simple connection
                us_reach = reaches[upstream_reaches_id[0]]

                us_reach_last_node = us_reach.nodes[us_reach.ds_nd_id]
                us_reach_last_node.dsid = ds_reach_us_junc_node.id
                us_reach_last_node.set_node_type(3)
                us_reach_last_node.us2id = -1

                # QUESTION: Does the ds_reach_us_junc_node need to be of a specific type? For now it will be "user defined" (default)
                ds_reach_us_junc_node.set_node_type(6)
                ds_reach_us_junc_node.usid = us_reach_last_node.id

                # print(f"\nUS Reach downstream node:")
                # print(us_reach_last_node)

                # print(f"\nDS Reach upstream node:")
                # print(ds_reach_us_junc_node)

            elif len(upstream_reaches_id) == 2:
                # go through the upstream topagnps reach ids in ascending id order, the first one will be the second inflow
                # according to cche1d numbering scheme of channels

                while upstream_reaches_id:
                    upstream_reach_id = upstream_reaches_id.pop(0)

                    us_reach = reaches[upstream_reach_id]

                    if upstream_reaches_id:  # the list is not empty
                        # us_reach = Second inflow
                        second_inflow_reach = us_reach
                        first_inflow_reach = reaches[
                            upstream_reaches_id[0]
                        ]  # the first inflow is the other one remaining

                    # make a copy of ds reach us_junc_node
                    latest_nd_id += 1
                    us_reach_ds_junc_node = Node(
                        id=latest_nd_id,
                        type=3,
                        usid=us_reach.ds_nd_id,
                        dsid=ds_reach_us_junc_node.id,
                        us2id=-1,
                        x=ds_reach_us_junc_node.x,
                        y=ds_reach_us_junc_node.y,
                        row=ds_reach_us_junc_node.row,
                        col=ds_reach_us_junc_node.col,
                    )

                    us_reach.add_node(us_reach_ds_junc_node)

                    # Make sure that the penultimate node (the one that used to be the ds_node
                    # actually points to the new one)
                    penultimate_node = us_reach.nodes[us_reach.ds_nd_id]
                    penultimate_node.dsid = latest_nd_id

                    # Update reach with ds_node information
                    us_reach.ds_nd_id = latest_nd_id

                ds_reach_us_junc_node.us2id = first_inflow_reach.ds_nd_id
                ds_reach_us_junc_node.usid = second_inflow_reach.ds_nd_id
                ds_reach_us_junc_node.set_node_type(
                    2
                )  # the junction node of two inflows is defined as type 2

    def _find_cells_inflow_nodes(self):
        """
        Find which node the cells are pouring into based on their type ('left', 'right', 'source')
        """

        cells = self.cells
        reaches = self.reaches

        for cell_id, cell in cells.items():
            receiving_reach_id = cell.receiving_reach_id
            receiving_reach = reaches[receiving_reach_id]

            if cell.type in ["left", "right"]:
                receiving_node_id = receiving_reach.ds_nd_id
            elif cell.type == "source":
                receiving_node_id = receiving_reach.us_nd_id

            # set also the property in the node itself
            receiving_node = receiving_reach.nodes[receiving_node_id]
            receiving_node.inflow_cell_sources.add(cell_id)

    def _find_removed_reaches_inflow_nodes(self):
        """
        If the network was simplified and reaches were removed then we need to set them as boundary conditions inflows
        """
        current_graph = self.current_graph
        full_graph = self.full_graph

        reaches = self.reaches

        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue

            current_graph_reach_predecessors = set(current_graph.predecessors(reach_id))
            full_graph_reach_predecessors = set(full_graph.predecessors(reach_id))

            removed_upstream_reaches = (
                full_graph_reach_predecessors - current_graph_reach_predecessors
            )

            reach_us_node = reach.nodes[reach.us_nd_id]
            # Get node immediately next
            second_most_us_node = reach.nodes[reach_us_node.dsid]

            second_most_us_node.inflow_reach_sources = set()

            if removed_upstream_reaches:
                if len(removed_upstream_reaches) == 1:
                    # only one reach is removed therefore the upstream node is an inflow
                    second_most_us_node.set_node_type(1)
                elif len(removed_upstream_reaches) > 1:
                    # more than one reach is removed therefore the upstream node is a source node
                    reach_us_node.set_node_type(0)


            for removed_reach in removed_upstream_reaches:
                reach_us_node.inflow_reach_sources.add(removed_reach)

    def _set_outlet_node_type(self):
        """
        Finds the outlet reach and sets its downstream node as type 9 for outlet
        """
        current_graph = self.current_graph

        for reach_id in current_graph:
            if current_graph.out_degree(reach_id) == 0:
                outlet_reach_id = reach_id
                break

        outlet_reach = self.reaches[outlet_reach_id]
        outlet_node_id = outlet_reach.ds_nd_id

        outlet_node = outlet_reach.nodes[outlet_node_id]
        outlet_node.set_node_type(9)

    def _set_inflow_nodes_type(self):
        """
        Finds the inflow reaches and set their respective upstream node as type 0 for source
        """
        current_graph = self.current_graph

        for reach_id in current_graph:
            if current_graph.in_degree(reach_id) == 0:
                inflow_reach = self.reaches[reach_id]
                inflow_node_id = inflow_reach.us_nd_id
                inflow_node = inflow_reach.nodes[inflow_node_id]
                inflow_node.set_node_type(0)

    def _set_default_node_type(self):
        """
        Goes through each node of each reach and set its type to default (6) unless it is already defined as something else
        """
        current_graph = self.current_graph

        reaches = self.reaches

        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            for node in reach.nodes.values():
                if node.type not in [0, 1, 2, 3, 9]:
                    node.set_node_type(6)

    def _create_reaches_from_df(self, df):
        """
        This function adds reaches to the watershed based on a DataFrame
        containing a "Reach_ID" and "Receiving_Reach" column
        """

        self.reaches = {}

        def convert_to_int_when_possible(x):
            try:
                return int(x)
            except:
                return x

        df["Reach_ID"] = df["Reach_ID"].apply(convert_to_int_when_possible)
        df["Receiving_Reach"] = df["Receiving_Reach"].apply(
            convert_to_int_when_possible
        )

        for _, row in df.iterrows():
            reach_id = row["Reach_ID"]
            receiving_reach_id = row["Receiving_Reach"]

            try:
                slope = row["Slope"]
            except:
                slope = None

            self.add_reach(
                Reach(id=reach_id, receiving_reach_id=receiving_reach_id, slope=slope)
            )

    def _create_cells_from_df(self, df):
        """
        This function adds cells to the watershed based on a DataFrame containing
        'Cell_ID', 'Reach_ID', 'Cell_Area' columns
        """

        self.cells = {}

        for _, row in df.iterrows():
            cell_id = int(row["Cell_ID"])
            reach_id = int(row["Reach_ID"])
            cell_area = row["Cell_Area"]
            self.add_cell(Cell(id=cell_id, receiving_reach_id=reach_id, area=cell_area))

    def _get_highest_node_id(self):
        max_nd_id = 0
        reaches = self.reaches
        # Remove type 3 nodes if they exist
        for reach_id, reach in reaches.items():
            if not reach.ignore:
                nodes = reach.nodes
                for node_id in nodes:
                    max_nd_id = max(max_nd_id, node_id)

        return max_nd_id

    def _get_node_byid(self, id, id_type="absolute"):
        """
        Go through reaches of watershed and find the node with provided id
        id_type:
            - 'absolute' corresponds to the "id" attribute
            - 'computeid' corresponds to the "computeid" attribute
        """
        nodes = self.nodes
        node = None
        if id_type.lower() == "absolute":
            return nodes[id]
        elif id_type.lower() == "computeid":
            for node in nodes.values():
                if node.computeid == id:
                    return node
        else:
            raise Exception(
                "Invalid id_type for selection of node. Valid values are 'absolute' and 'computeid'"
            )

        if node is None:
            raise Exception(f"No valid none found for id: {id} and id_type: {id_type}")

    def _get_highest_cross_section_id(self):
        max_cs_id = 0
        crosss_sections = self.cross_sections
        # Remove type 3 nodes if they exist
        for c_section in crosss_sections.values():
            max_cs_id = max(c_section.id, max_cs_id)

        return max_cs_id

    def _print_reaches_node_ids(self, computeidnone=True):
        reaches = self.reaches

        for reach in reaches.values():
            if reach.ignore:
                continue
            for node in reach.nodes.values():
                if node.computeid is None or not (computeidnone):
                    print(
                        f"Reach {reach.id}, Node: {node.id}, ComputeID: {node.computeid}"
                    )
