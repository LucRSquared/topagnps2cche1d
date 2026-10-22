import warnings
from pathlib import Path
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
from topagnps2cche1d.network import (
    build_network_from_reaches,
    write_cche1d_dat_files as _write_cche1d_dat_files,
    validate_network,
    inflow_reaches_records,
)

import holoviews as hv
import hvplot.pandas


class Watershed:
    """
    TopAGNPS import layer.

    A ``Watershed`` holds the *generic* topology read from a TopAGNPS dataset:
    reaches (id + receiving_reach_id + a US->DS chain of ``Node``s with row/col
    and x/y coordinates), cells, and per-node cross sections, plus the
    reach-connectivity graph.

    The CCHE1D channel network (nodes / channels / links / reaches / cross
    sections) is *not* stored here.  It is derived on demand by
    :meth:`build_cche1d_network`, which delegates to
    ``topagnps2cche1d.network.build_network_from_reaches``.  That builder is
    the single source of truth for the CCHE1D node typing, junction block and
    computing order, and for writing the six ``.dat`` input files.  Keeping it
    in one place means a hand-defined network and an imported one are
    indistinguishable to the CCHE1D Fortran reader.
    """

    def __init__(self):
        self.reaches = {}
        self.cells = {}
        self.cross_sections = {}
        self.full_graph = None
        self.current_graph = None

        self.geomatrix = None

        # CCHE1D network cache (a network.Cche1dNetwork), built on demand.
        self._cche1d_network = None
        self._network_built_at_node_id = None
        self._network_strahler_drop = None

    # ------------------------------------------------------------------
    # Generic topology: import, graph, geometry, orientation
    # ------------------------------------------------------------------

    def add_reach(self, reach):
        self.reaches[reach.id] = reach

    def add_cell(self, cell):
        self.cells[cell.id] = cell

    def add_cross_section(self, cross_section):
        self.cross_sections[cross_section.id] = cross_section

    def import_topagnps_reaches_network(self, path_reach_data_section):
        """
        Imports information from "AnnAGNPS_Reach_Data_Section.csv" file produced
        by TopAGNPS and adds it to the Watershed.
        """
        df = read_reach_data_section(path_reach_data_section)

        self._create_reaches_from_df(df)

        self.update_graph()

        self.assign_strahler_number_to_reaches()

    def import_topagnps_cells(self, path_cell_data_section):
        """
        Imports information from "AnnAGNPS_Cell_Data_Section.csv" file produced
        by TopAGNPS and adds it to the Watershed.
        """

        df = read_cell_data_section(path_cell_data_section)
        df["Cell_Area"] = (
            df["Cell_Area"] * 1e4
        )  # TopAGNPS produces an area in ha, we want it in m²

        self._create_cells_from_df(df)

    def update_graph(self):
        """
        Replaces the connectivity by NetworkX DiGraphs.
        A fully connected graph (``full_graph``) is built from every reach, and
        a subgraph (``current_graph``) is built from the reaches that are not
        ignored.
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
        Prepare the *generic* topology for CCHE1D build:

            - rebuild the connectivity graph from the current ``ignore`` flags
            - give every single-node reach its upstream and downstream nodes

        The CCHE1D-specific work (junction detection, node typing, computing
        order, node renumbering) is intentionally **not** done here.  It is
        performed by :meth:`build_cche1d_network`, which derives it from the
        generic chains and connectivity.  Call this method whenever the
        topology changes (after ``ignore_reaches_*`` or ``resample_reaches``).
        """
        self.update_graph()

        if self.current_graph is None or len(self.current_graph) == 0:
            raise Exception(
                "No reaches left to consider in Watershed connectivity graph"
            )

        # Identify reaches with a single node and add two extra nodes, one
        # upstream and one downstream, so every kept reach has a complete
        # US->DS chain for the CCHE1D builder.
        self.add_us_and_ds_node_to_single_node_reaches()

    def keep_all_reaches(self):
        """
        This function sets all the reaches in the watershed to non ignored
        """
        for reach in self.reaches.values():
            reach.ignore = False

    def ignore_reaches_with_strahler_leq(self, strahler_threshold=0):
        """
        Ignore reaches with Strahler Number less than or equal to a given threshold
        """
        self.keep_all_reaches()
        for reach in self.reaches.values():
            if reach.strahler_number <= strahler_threshold:
                reach.ignore_reach()

        self.update_graph()

    def assign_strahler_number_to_reaches(self, mode="fully_connected"):
        """
        Assign Strahler number to reaches in the system according to the connectivity dict.
        mode: 'fully_connected' (default) -> Uses the fully connected network
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
        highest_strahler = 0
        for reach in self.reaches.values():
            if reach.strahler_number is not None and reach.strahler_number > highest_strahler:
                highest_strahler = reach.strahler_number
        return highest_strahler

    # ------------------------------------------------------------------
    # Geometry (raster -> node coordinates) and orientation
    # ------------------------------------------------------------------

    def read_reaches_geometry_from_topagnps_asc_file(self, path_reach_raster):
        """
        This function reads the reach geometry from the raster provided as input,
        it will set the row and col coordinate of all the nodes in the watershed.
        ### Parameter:
            - path_reach_raster: Path to the reach geometry raster (.asc) file produced by TopAGNPS
        """
        reaches = self.reaches
        from tqdm import tqdm

        img_reach_asc, geomatrix, _, _, nodataval_reach_asc, _ = read_esri_asc_file(
            path_reach_raster
        )

        self.geomatrix = geomatrix

        # Get list of reach ids that are present in the raster
        mask_no_data = img_reach_asc == nodataval_reach_asc
        list_of_reaches_in_raster = set(
            np.unique(img_reach_asc[~mask_no_data]).astype(int)
        )

        nd_counter = 0
        for reach_id in tqdm(list_of_reaches_in_raster, desc="Reading Reaches"):
            reach_img = np.where(img_reach_asc == reach_id, 1, 0)

            connected_pixels_graph = get_pathgraph_binary(
                reach_img, output_index_starts_one=False
            )
            extremities = find_extremities_pathgraph(connected_pixels_graph)

            if len(extremities) == 1:
                # Reach is a single pixel: one node, both ends coincide
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

            # 1-indexed: k=1 is the start node, k=N is the end node
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

    def add_us_and_ds_node_to_single_node_reaches(self):
        """
        Looks for single node reaches and adds an upstream and downstream node to them

        Idempotent: a reach that already has 2+ nodes is left untouched, so this
        method can safely be re-run inside repeated ``update_watershed`` calls.
        The new endpoints sit at the midpoint between the single node and the
        (kept) neighbour's connecting node, mirroring the geometry the CCHE1D
        builder gives zero-length segments.
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
            if len(reach_middle.nodes) == 1:
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
                        x=(node_reach_before_ds.x + node_middle.x) / 2,
                        y=(node_reach_before_ds.y + node_middle.y) / 2,
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
                        x=(node_reach_after_us.x + node_middle.x) / 2,
                        y=(node_reach_after_us.y + node_middle.y) / 2,
                    )
                )
                reach_middle.ds_nd_id = nd_counter

                node_middle.dsid = nd_counter

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

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

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

        # Topology changed: rebuild the graph and single-node expansion.
        self.update_watershed()
        # The cached CCHE1D network is now stale (node ids changed).
        self._cche1d_network = None

    # ------------------------------------------------------------------
    # Inflow sources (cells and removed reaches)
    # ------------------------------------------------------------------

    def identify_inflow_sources(self):
        """
        There are two types of inflow sources:
            1. Cells all throughout the watershed pouring into either source nodes
               or the downstream node of their adjacent reach
            2. Inflow from a "ghost" reach that was removed
        """
        self._find_cells_inflow_nodes()
        self._find_removed_reaches_inflow_nodes()

    def _find_cells_inflow_nodes(self):
        """
        Find which node the cells are pouring into based on their type ('left', 'right', 'source')
        """
        cells = self.cells
        reaches = self.reaches

        for cell_id, cell in cells.items():
            receiving_reach_id = cell.receiving_reach_id
            if receiving_reach_id not in reaches:
                continue
            receiving_reach = reaches[receiving_reach_id]

            if cell.type in ["left", "right"]:
                receiving_node_id = receiving_reach.ds_nd_id
            elif cell.type == "source":
                receiving_node_id = receiving_reach.us_nd_id
            else:
                continue

            # set also the property in the node itself
            receiving_node = receiving_reach.nodes[receiving_node_id]
            receiving_node.inflow_cell_sources.add(cell_id)

    def _find_removed_reaches_inflow_nodes(self):
        """
        Stamp the head node of each kept reach with the ids of the removed
        upstream reaches it receives (``Node.inflow_reach_sources``).  This is
        the generic-layer bookkeeping, used by ``Node.__str__`` and plotting.
        The head node is the receiving boundary: it is the CCHE1D inflow node
        (type 1 or 0) that ``build_network_from_reaches`` classifies, so the
        stamp, the builder's head typing, and the model-facing record
        (``network.inflow_reaches_records``) all point at the same node.
        """
        current_graph = self.current_graph
        full_graph = self.full_graph
        reaches = self.reaches

        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            removed_upstream_reaches = (
                set(full_graph.predecessors(reach_id))
                - set(current_graph.predecessors(reach_id))
            )
            if not removed_upstream_reaches:
                continue
            head = reach.nodes[reach.us_nd_id]
            head.inflow_reach_sources = set(removed_upstream_reaches)
            # display node type: one removed inflow -> inflow boundary (1),
            # several -> source (0)
            head.set_node_type(1 if len(removed_upstream_reaches) == 1 else 0)

    # ------------------------------------------------------------------
    # Cross sections (per generic node)
    # ------------------------------------------------------------------

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
        /!\\ these indexes start at 1 (ONE) not 0 (ZERO)
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

        # Find the outlet reach: the one that no reach flows into.
        current_graph = self.current_graph
        outlet_reach_id = None
        for rid in current_graph:
            if current_graph.out_degree(rid) == 0:
                outlet_reach_id = rid
                break
        if outlet_reach_id is None:
            raise Exception("Could not determine the outlet reach")

        # Get list of reaches from downstream to upstream in a Breadth-First-Search order
        reaches_to_visit_going_upstream = list(
            nx.bfs_tree(self.current_graph.reverse(), outlet_reach_id)
        )

        outlet_reach = reaches[outlet_reach_id]
        outlet_node = outlet_reach.nodes[outlet_reach.ds_nd_id]

        for reach_id in reaches_to_visit_going_upstream:
            # Get reach
            current_reach = reaches[reach_id]
            # Identify the ID of its DS and US nodes
            ds_nd_id = current_reach.ds_nd_id
            us_nd_id = current_reach.us_nd_id

            # Node chain from DS to US (reverse of the US->DS chain).  The
            # chain is derived from the node order, not from per-node usid
            # links, so reaches whose nodes never got a usid (hand-built or
            # resampled interiors) work the same as raster-read ones.
            forward = [us_nd_id]
            cur = us_nd_id
            while cur != ds_nd_id:
                cur = current_reach.nodes[cur].dsid
                forward.append(cur)
            chain = list(reversed(forward))

            if ds_nd_id == outlet_node.id:
                # first reach: the outlet node's own elevation is the
                # reference (zero-distance first step), as in the reference
                # pipeline
                previous_node = current_reach.nodes[ds_nd_id]
                elevation = outlet_elevation  # Initial elevation
            else:
                ds_reach = reaches[current_reach.receiving_reach_id]
                # Restart elevation at the elevation of the previous node
                previous_node = ds_reach.nodes[ds_reach.us_nd_id]
                elevation = cross_sections[previous_node.csid].get_thalweg_elevation()

            for node_id in chain:
                # Compute distance from previous node and the resulting
                # elevation of the current node (elevation += slope * dist)
                current_node = current_reach.nodes[node_id]
                dist_prev_node = current_node.distance_from(previous_node)
                elevation += dist_prev_node * current_reach.slope
                # Shift current node cross section to the absolute elevation
                cross_section = cross_sections[current_node.csid]
                cross_section.set_thalweg_elevation(elevation)
                previous_node = current_node

    # ------------------------------------------------------------------
    # CCHE1D network (delegated to topagnps2cche1d.network)
    # ------------------------------------------------------------------

    def build_cche1d_network(self, strahler_drop=0, resample_step=None,
                             validate=True):
        """
        Build (and cache) the CCHE1D channel network for the current topology.

        Delegates to :func:`topagnps2cche1d.network.build_network_from_reaches`,
        which derives node typing, the junction block and the CCHE1D computing
        order from the generic reach chains and the ``receiving_reach_id``
        connectivity.  Every node is given the per-node :class:`CrossSection`
        currently assigned to it (if any).

        Returns a ``network.Cche1dNetwork``.  The result is cached and reused
        until the topology changes (a later :meth:`resample_reaches` invalidates
        it).  Call :meth:`update_watershed` first after any topological change.
        """
        self.update_watershed()

        # Populate cell / removed-reach inflow sources.  The tw-cells and
        # inflow-reaches DataFrames read these, and they only make sense once the
        # topology (graphs + single-node expansion) is final.
        self.identify_inflow_sources()

        # Per-node cross sections keyed by generic node id.  The builder maps
        # each CCHE1D node back to its generic node (node_provenance) and
        # assigns the matching section.  Only active (kept) reaches matter.
        cs_by_generic_id = {}
        for reach_id in self.current_graph:
            for node in self.reaches[reach_id].nodes.values():
                if node.csid is not None and node.csid in self.cross_sections:
                    cs_by_generic_id[node.id] = self.cross_sections[node.csid]

        # Pass the FULL reach set (kept + removed).  The builder keeps the
        # reaches that are not ``ignore`` and above the Strahler threshold, but
        # it needs the removed reaches too so a kept reach's head can be
        # classified (one removed inflow -> inflow boundary, several -> source).
        net = build_network_from_reaches(
            list(self.reaches.values()),
            strahler_drop=strahler_drop,
            resample_step=resample_step,
            cs_by_generic_id=cs_by_generic_id or None,
        )

        if validate:
            errors, _warnings = validate_network(net)
            if errors:
                raise ValueError(
                    f"Built CCHE1D network failed validation: {errors}"
                )

        self._cche1d_network = net
        self._network_built_at_node_id = self._get_highest_node_id()
        self._network_strahler_drop = strahler_drop
        return net

    def get_cche1d_network(self, strahler_drop=0, resample_step=None):
        """
        Return the cached CCHE1D network, building it if needed or stale.
        """
        if (
            self._cche1d_network is None
            or self._network_stale()
            or self._network_strahler_drop != strahler_drop
        ):
            return self.build_cche1d_network(
                strahler_drop=strahler_drop, resample_step=resample_step
            )
        return self._cche1d_network

    def _network_stale(self):
        return (
            self._network_built_at_node_id is None
            or self._get_highest_node_id() != self._network_built_at_node_id
        )

    # ------------------------------------------------------------------
    # CCHE1D DataFrames (read off the built network)
    # ------------------------------------------------------------------

    def create_cche1d_nodes_df(self):
        """
        Generate a DataFrame containing all the nodes in CCHE1D format (12
        columns), derived from the built CCHE1D network.
        """
        net = self.get_cche1d_network()
        rows = [
            (n.nd_id, n.frmno, n.nd_type, n.xc, n.yc, n.dsid, n.usid, n.us2id,
             n.csid, n.rsid, n.stid, n.sbid)
            for n in sorted(net.nodes, key=lambda n: n.nd_id)
        ]
        df = pd.DataFrame(
            rows,
            columns=["ND_ID", "ND_FRMNO", "ND_TYPE", "ND_XC", "ND_YC",
                     "ND_DSID", "ND_USID", "ND_US2ID", "ND_CSID", "ND_RSID",
                     "ND_STID", "ND_SBID"],
        )
        return df

    def create_cche1d_channels_df(self):
        """
        Generate a DataFrame containing all the channels in CCHE1D format.
        """
        net = self.get_cche1d_network()
        df = pd.DataFrame(
            [(c.ch_id, c.ndusid, c.nddsid)
             for c in sorted(net.channels, key=lambda c: c.ch_id)],
            columns=["CH_ID", "CH_NDUSID", "CH_NDDSID"],
        )
        return df

    def create_cche1d_links_and_reaches_df(self):
        """
        Generate DataFrames containing all the links and reaches in CCHE1D
        format.
        /!\\ A CCHE1D reach is not the same as a TopAGNPS reach. A CCHE1D reach is
        simply a section between two nodes.
        """
        net = self.get_cche1d_network()

        df_lk = pd.DataFrame(
            [(l.lk_id, l.cmpseq, l.ndusid, l.nddsid, l.rcusid, l.rcdsid,
              l.lk_type)
             for l in sorted(net.links, key=lambda l: l.lk_id)],
            columns=["LK_ID", "LK_CMPSEQ", "LK_NDUSID", "LK_NDDSID",
                     "LK_RCUSID", "LK_RCDSID", "LK_TYPE"],
        )

        df_rc = pd.DataFrame(
            [(r.rc_id, r.ndusid, r.nddsid, r.order, r.length)
             for r in sorted(net.reaches, key=lambda r: r.rc_id)],
            columns=["RC_ID", "RC_NDUSID", "RC_NDDSID", "RC_ORDER", "RC_LENGTH"],
        )

        return df_lk, df_rc

    def create_cche1d_csec_csprf_df(self):
        """
        Create the two DataFrames that describe the cross sections, derived
        from the built CCHE1D network.
        """
        net = self.get_cche1d_network()

        cs_ID, cs_NPTS, cs_LOB, cs_LBT, cs_ROB, cs_RBT = [], [], [], [], [], []
        cs_SVTYPE, cs_TYPE, cs_ORIGIN, cs_STATION = [], [], [], []

        cp_ID, cp_CSID, cp_POSIDX = [], [], []
        cp_W, cp_Z, cp_RGH, cp_BLOCK = [], [], [], []

        cp_id = 0
        for cs in sorted(net.csecs, key=lambda c: c.cs_id):
            points = sorted(cs.points, key=lambda p: p[1])
            cp_id += len(points)
            cp_ID.extend(range(cp_id - len(points) + 1, cp_id + 1))
            cp_CSID.extend([cs.cs_id for _ in points])
            for (posidx, w, z, rgh, block) in points:
                cp_POSIDX.append(posidx)
                cp_W.append(w)
                cp_Z.append(z)
                cp_RGH.append(rgh)
                cp_BLOCK.append(block)

            cs_ID.append(cs.cs_id)
            cs_NPTS.append(len(cs.points))
            cs_LOB.append(cs.lob)
            cs_LBT.append(cs.lbt)
            cs_ROB.append(cs.rob)
            cs_RBT.append(cs.rbt)
            cs_SVTYPE.append("WZ")
            cs_TYPE.append("MC")
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

        return df_csec, df_csprf

    def create_cche1d_twcells_df(self):
        """
        Create a DataFrame of the cells pouring into the CCHE1D reach network.
        Node references are the CCHE1D network node ids (ND_ID).

        A cell drains into the generic node it is recorded against, which maps
        onto the network node via the bijective origin map (``node_origin``):
        even a junction-tail network node (type 3) keeps the cell drainage of
        the generic node it was created from.
        """
        net = self.get_cche1d_network()
        # generic node id -> network node id (bijective for kept nodes)
        inv_origin = {
            gid: nid for nid, gid in net.node_origin.items() if gid is not None
        }
        # network node id -> its upstream node id (for convenience column)
        net_usid = {n.nd_id: n.usid for n in net.nodes}

        cells = self.cells
        reaches = self.reaches
        current_graph = self.current_graph

        tw_ID, tw_NUMB, tw_NDDSID, tw_NDUSID, tw_DRAREA = [], [], [], [], []
        tw_id = 0
        for reach_id, reach in reaches.items():
            if reach_id not in current_graph:
                continue
            for node in reach.nodes.values():
                for cell_source_id in node.inflow_cell_sources:
                    tw_id += 1
                    cell_source = cells[cell_source_id]
                    net_nd = inv_origin.get(node.id)
                    if net_nd is None:
                        raise KeyError(
                            f"cell {cell_source_id} drains into node "
                            f"{node.id}, which has no CCHE1D network node")
                    tw_ID.append(tw_id)
                    tw_NUMB.append(cell_source_id)
                    tw_NDDSID.append(net_nd)
                    tw_NDUSID.append(net_usid.get(net_nd, -1))
                    tw_DRAREA.append(cell_source.area)

        return pd.DataFrame(
            {
                "TW_ID": tw_ID,
                "TW_NUMB": tw_NUMB,
                "TW_NDDSID": tw_NDDSID,
                "TW_NDUSID": tw_NDUSID,
                "TW_DRAREA": tw_DRAREA,
            }
        )

    def get_list_of_inflow_reaches_df(self):
        """
        Create a DataFrame with a list of reaches that were removed from the
        network (coarsened away) and the CCHE1D node each one feeds.  Node
        references are CCHE1D network node ids (``ND_ID``), specifically the
        **head** node of the kept reach the removed reach discharges into — the
        inflow boundary condition node.

        The record is derived by :func:`topagnps2cche1d.network.inflow_reaches_records`
        from the built network's ``removed_reaches`` / ``channel_heads`` (the
        same source-agnostic implementation a hand-built network uses), so the
        TopAGNPS and manual paths can never disagree.  This is the list of
        AnnAGNPS reach outputs that must be generated and wired in as CCHE1D
        inflow boundary conditions.
        """
        net = self.get_cche1d_network()
        rows = inflow_reaches_records(net)
        df = pd.DataFrame(
            {
                "inflow_topagnps_reach_id": [r for r, _ in rows],
                "receiving_node_id": [n for _, n in rows],
            }
        )
        if not df.empty:
            df = df.sort_values(by="inflow_topagnps_reach_id")
            df.reset_index(inplace=True, drop=True)
        return df

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_cche1d_dat_files(
        self, casename="topagnps", output_folder=None, float_format="%.4f", sep="\t"
    ):
        """
        Generate all the CCHE1D dat files in the specified folder.
        The six core files (channel / csec / csprf / link / nodes / reach) are
        written by :func:`topagnps2cche1d.network.write_cche1d_dat_files` from
        the built network; in addition this writes:
            * {casename}_tw_cell.dat
            * {casename}_inflow_reaches.csv (only if non-empty)
        to know which reach outputs need to be generated in AnnAGNPS if some of
        them were removed from the network to provide correct inflow BCs.
        """
        # Topology may have changed since the last build; rebuild on demand.
        self.get_cche1d_network()

        # Core six files (nodes / reach / channel / link / csec / csprf).
        paths = _write_cche1d_dat_files(self._cche1d_network, output_folder or ".",
                                        prefix=casename)
        if output_folder is None:
            output_folder = Path().cwd()
        else:
            output_folder = Path(output_folder)

        def _write_df(df, filename):
            filename.write_text(f"{df.shape[0]}\n")
            df.to_csv(filename, mode="a", index=False,
                      float_format=float_format, sep=sep)

        df_tw = self.create_cche1d_twcells_df()
        file_tw = Path(output_folder) / f"{casename}_tw_cell.dat"
        _write_df(df_tw, file_tw)

        df_inflow_reaches = self.get_list_of_inflow_reaches_df()
        file_inflow_reaches = Path(output_folder) / f"{casename}_inflow_reaches.csv"
        if df_inflow_reaches.size != 0:
            df_inflow_reaches.to_csv(file_inflow_reaches, index=False)

        result = {
            "nodes": paths["nodes"],
            "reach": paths["reach"],
            "channel": paths["channel"],
            "link": paths["link"],
            "csec": paths["csec"],
            "csprf": paths["csprf"],
            "tw_cell": file_tw,
        }
        if df_inflow_reaches.size != 0:
            result["inflow_reaches"] = file_inflow_reaches
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, **kwargs):
        """
        Plot the generic watershed network (all kept reaches) with holoviews.
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
        dfs = dfs.sort_values(by="ID")

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
                    "DSID",
                    "Reach_ID",
                    "Strahler_Number",
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
                    "DSID",
                    "Reach_ID",
                    "Strahler_Number",
                ],
            )
        ).opts(
            frame_width=frame_width,
            frame_height=frame_height,
            aspect=aspect,
            title=title,
        )

        return watershed_plot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_reaches_from_df(self, df):
        """
        This function adds reaches to the watershed based on a DataFrame
        containing a "Reach_ID" and "Receiving_Reach" column
        """
        self.reaches = {}

        def convert_to_int_when_possible(x):
            try:
                return int(x)
            except Exception:
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
            except Exception:
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
            self.add_cell(
                Cell(id=cell_id, receiving_reach_id=reach_id, area=cell_area)
            )

    def _get_highest_node_id(self):
        max_nd_id = 0
        for reach in self.reaches.values():
            for node_id in reach.nodes:
                max_nd_id = max(max_nd_id, node_id)
        return max_nd_id

    def _get_node_byid(self, id, id_type="absolute"):
        """
        Go through reaches of watershed and find the node with provided id
        id_type:
            - 'absolute' corresponds to the "id" attribute
            - 'computeid' corresponds to the "computeid" attribute
        """
        if id_type.lower() == "absolute":
            for reach in self.reaches.values():
                if id in reach.nodes:
                    return reach.nodes[id]
            raise Exception(f"No node with id {id} found in watershed")
        elif id_type.lower() == "computeid":
            for reach in self.reaches.values():
                for node in reach.nodes.values():
                    if node.computeid == id:
                        return node
            raise Exception(f"No node with computeid {id} found in watershed")
        else:
            raise Exception(
                "Invalid id_type for selection of node. Valid values are 'absolute' and 'computeid'"
            )

    def _get_highest_cross_section_id(self):
        max_cs_id = 0
        for c_section in self.cross_sections.values():
            max_cs_id = max(c_section.id, max_cs_id)
        return max_cs_id

    def _print_reaches_node_ids(self, computeidnone=True):
        for reach in self.reaches.values():
            if reach.ignore:
                continue
            for node in reach.nodes.values():
                if node.computeid is None or not (computeidnone):
                    print(
                        f"Reach {reach.id}, Node: {node.id}, ComputeID: {node.computeid}"
                    )
