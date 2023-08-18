import warnings
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
from topagnps2cche1d.tools import (
    custom_dfs_traversal_sorted_predecessors,
    read_esri_asc_file,
    read_reach_data_section,
    read_cell_data_section,
    find_extremities_binary,
    get_intermediate_nodes_img,
)
from topagnps2cche1d import Reach, Node, Cell

import holoviews as hv
import hvplot.pandas


class Watershed:
    def __init__(self):
        self.reaches = {}
        self.cells = {}
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
        """Replaces the connectivity dict by a NetworkX DiGraph
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

    def read_reaches_geometry_from_polygons_gdf(gdf, column_id_name="dn"):
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
            extremities = find_extremities_binary(
                reach_img, output_index_starts_one=False
            )

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

            # TODO: optimize with path_crawler
            reach_skel_rowcols = get_intermediate_nodes_img(
                startrowcol, endrowcol, reach_img
            )

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
        This function should be called AFTER identify_inflow_sources so that node types are correctly
        identified
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

            while True:
                computeid += 1
                current_node = nodes[current_node_id]
                current_node.computeid = computeid

                if current_node_id == ds_node_id:
                    break
                else:
                    current_node_id = current_node.dsid

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

    def plot(self, **kwargs):
        """
        Plot network
        """
        reaches_plots = []

        if 'aspect' in kwargs:
            aspect = kwargs['aspect']
        else:
            aspect = 'equal'

        if 'frame_width' in kwargs:
            frame_width = kwargs['frame_width']
        else:
            frame_width = 1000

        if 'frame_height' in kwargs:
            frame_height = kwargs['frame_height']
        else:
            frame_height = 1000

        if 'by' in kwargs:
            by = kwargs['by']
        else:
            by = 'Reach_ID'


        for reach in self.reaches.values():
            reaches_plots.append(reach.get_nodes_as_df())

        dfs = pd.concat(reaches_plots, ignore_index=True)

        watershed_plot = (
            dfs.hvplot(x='X', y='Y', by=by, kind='line',
                        hover_cols=['TYPE', 'US2ID', 'USID', 'ID', 'COMPUTEID', 'DSID']) * \
            dfs.hvplot(x='X', y='Y', by=by, kind='scatter', alpha=0.5,
                        hover_cols=['TYPE', 'US2ID', 'USID', 'ID', 'COMPUTEID', 'DSID'])
        ).opts(frame_width=frame_width, frame_height=frame_height, aspect=aspect) 

        return watershed_plot 
        

    def _create_junctions_between_reaches(self):
        """
        Every reach that has exactly two upstream reaches needs to have its most upstream node added as the most downstream node
        (= 'end of link' node with type = 3). The DSID, USID, US2ID need to be updated accordingly.
        In two inflows junction, the second inflow (which is the last reach upstream of the downstream reach when numbered in a depth
        first search method with a right hand rule (looking upstream)). The second inflow is thus also the reach with the smallest reach_id
        value according to TopAGNPS numbering system.

        If a reach has only one it's a trivial connection and the DSID, USID, and TYPE can be easily handled.
        """

        latest_nd_id = self._get_highest_node_id()
        reaches = self.reaches
        current_graph = self.current_graph

        # Remove type 3 nodes if they exist
        for reach in reaches.values():
            nodes = reach.nodes
            for node_id, node in nodes.items():
                if node.type == 3:
                    usid = node.usid
                    reach.ds_nd_id = (
                        usid  # the new ds_nd_id is the node that was immediately before
                    )
                    nodes[
                        usid
                    ].dsid = None  # the node immediately before now has no dsid
                    del nodes[node_id]  # delete the node of type 2
                if node.type == 2:
                    node.type = None

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

                    # make a copy of us_junc_node
                    latest_nd_id += 1
                    us_reach_ds_junc_node = Node(
                        id=latest_nd_id,
                        type=3,
                        usid=us_reach.ds_nd_id,
                        dsid=reach.us_nd_id,
                        us2id=-1,
                        x=ds_reach_us_junc_node.x,
                        y=ds_reach_us_junc_node.y,
                        row=ds_reach_us_junc_node.row,
                        col=ds_reach_us_junc_node.col,
                    )

                    us_reach.add_node(us_reach_ds_junc_node)

                    # Make sure that the penultimate node (the one that used to be the ds_node
                    # actually points to the new one)
                    penultimate_node = us_reach.nodes[us_reach_ds_junc_node.usid]
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
            receiving_node.inflow_cell_source = cell_id

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
            reach_us_node.inflow_reaches_source = []

            if removed_upstream_reaches:
                reach_us_node.set_node_type(1)

            for removed_reach in removed_upstream_reaches:
                reach_us_node.inflow_reaches_source.append(removed_reach)

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
        reaches = self.reaches

        for reach in reaches.values():
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
            nodes = reach.nodes
            for node_id in nodes:
                max_nd_id = max(max_nd_id, node_id)

        return max_nd_id