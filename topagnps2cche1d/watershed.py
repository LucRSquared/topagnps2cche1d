import warnings
import copy
import networkx as nx
from topagnps2cche1d.tools import (
    custom_dfs_traversal_sorted_predecessors,
    read_esri_asc_file,
    read_agflow_reach_data,
    read_reach_data_section,
)
from topagnps2cche1d import Reach
import pandas as pd


class Watershed:
    def __init__(self):
        self.reaches = {}
        self.full_graph = None
        self.current_graph = None

    def add_reach(self, reach):
        self.reaches[reach.id] = reach

    def _create_reaches_from_df(self, df):
        """
        This function add reaches to the watershed based on a DataFrame
        containing a "Reach_ID" and "Receiving_Reach" column
        """

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

        # Find reaches that are in the Receiving_Reach column but not in the Reach_ID column
        # and add them to the collection
        only_receiving_reaches = set(df["Receiving_Reach"].to_list()) - set(
            df["Reach_ID"].to_list()
        )
        for only_receiving_reach in only_receiving_reaches:
            self.add_reach(Reach(id=only_receiving_reach))

    def import_topagnps_reaches_network(self, path_reach_data_section):
        """
        Imports information from "AnnAGNPS_Reach_Data_Section.csv" file produced by TopAGNPS
        and adds to Watershed
        """
        df = read_reach_data_section(path_reach_data_section)

        self._create_reaches_from_df(df)

        self.update_graph()

        self.assign_strahler_number_to_reaches()

    def update_graph(self):
        """Replaces the connectivity dict by a NetworkX DiGraph
        Creates a fully connected graph from all the reaches that are in the watershed
        And then creates a subgraph -- current_graph from the reaches that are not ignored
        """
        reaches = self.reaches

        full_graph = nx.DiGraph()

        edges_full = []
        for reach_id, reach in reaches.items():
            if reach.receiving_reach_id is None:
                continue
            edges_full.append((reach_id, reach.receiving_reach_id))

        full_graph.add_edges_from(edges_full)

        # current_graph = full_graph.copy()
        ignored_reaches = set()
        for reach_id, reach in reaches.items():
            if reach.ignore:
                # set all upstream reaches to ignore
                for ancestor_reach_id in nx.ancestors(full_graph, reach_id):
                    reaches[ancestor_reach_id].ignore_reach()
                    ignored_reaches.add(ancestor_reach_id)

        kept_reaches = set(full_graph.nodes) - ignored_reaches
        current_graph = full_graph.subgraph(kept_reaches)

        self.full_graph = full_graph
        self.current_graph = current_graph

    def get_ignored_reaches_id(self):
        reaches = self.reaches
        ignored_reaches = []
        for id, reach in reaches.items():
            if reach.ignore:
                ignored_reaches.append(id)

        return ignored_reaches

    def keep_all_reaches(self):
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
        # queue = list(connectivity_dict.keys())

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
