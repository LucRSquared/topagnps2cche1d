import warnings
import copy
from topagnps2cche1d.tools import dfs_iterative_postorder
from topagnps2cche1d import Reach
import pandas as pd

class Watershed:
    def __init__(self):
        self.reaches = {}
        self.all_reaches_connectivity_dict = {}
        self.current_connectivity_dict = {}

    def add_reach(self, reach):
        self.reaches[reach.id] = reach

    def _read_reach_data_section(self, path_reach_data_section):
        """
        This function reads an "AnnAGNPS_Reach_Data_Section.csv" file produced by TopAGNPS
        """
        return pd.read_csv(path_reach_data_section)
    
    def _create_reaches_from_df(self, df):
        """
        This function add reaches to the watershed based on a DataFrame
        containing a "Reach_ID" and "Receiving_Reach" column 
        """
        for _, row in df.iterrows():
            reach_id           = row['Reach_ID']
            receiving_reach_id = row['Receiving_Reach']
            try:
                slope = row['Slope']
            except:
                slope = None
                
            self.add_reach(Reach(id=reach_id,
                                 receiving_reach_id=receiving_reach_id,
                                 slope=slope))
            
    def import_topagnps_reaches_network(self, path_reach_data_section):
        """
        Imports information from "AnnAGNPS_Reach_Data_Section.csv" file produced by TopAGNPS
        and adds to Watershed
        """
        df = self._read_reach_data_section(path_reach_data_section)
        self._create_reaches_from_df(df)
        self.update_connectivity()

        # Find root_id because that reach is missing when importing df
        root_id = self._find_root_reach(self.all_reaches_connectivity_dict)

        if root_id not in self.reaches:
            self.add_reach(Reach(id=root_id))

        self.assign_strahler_number_to_reaches()


    def _create_connectivity_dict(self):
        connectivity_dict = {}

        for reach_id, reach in self.reaches.items():
            receiving_reach_id = reach.receiving_reach_id

            if receiving_reach_id not in connectivity_dict:
                connectivity_dict[receiving_reach_id] = [reach_id]
            else:
                connectivity_dict[receiving_reach_id].append(reach_id)
        try:
            del connectivity_dict[None]
        except:
            pass
        
        return connectivity_dict

    def _find_ignored_reaches(self, connectivity_dict):
        ignored_reaches = set()
        for reach_id, reach in self.reaches.items():
            if reach.ignore:
                ignored_reaches.add(reach_id)

        upstream_of_ignored_reaches = set()
        if ignored_reaches:
            for elem in ignored_reaches:
                if elem in connectivity_dict:
                    upstream_of_ignored_reaches.update(connectivity_dict[elem])

        return ignored_reaches, upstream_of_ignored_reaches

    def _set_ignored_reaches(self, connectivity_dict, ignored_reaches, upstream_of_ignored_reaches):
        all_ignored_reaches = ignored_reaches.union(upstream_of_ignored_reaches)

        for reach_id, reach in self.reaches.items():
            if reach_id in all_ignored_reaches:
                reach.ignore = True
                if reach_id in connectivity_dict:
                    del connectivity_dict[reach_id]
                for dict_reach_id, dict_reaches_upstream in connectivity_dict.items():
                    if reach_id in dict_reaches_upstream:
                        dict_reaches_upstream.remove(reach_id)

    def _find_root_reach(self, connectivity_dict):

        all_upstream_reaches = []

        for upstream_reaches in connectivity_dict.values():
            all_upstream_reaches.extend(upstream_reaches)

        for reach_id in connectivity_dict:
            if reach_id not in all_upstream_reaches:
                return reach_id
            
    def update_connectivity(self):
        # Create a fully connected network
        all_reaches_connectivity_dict = self._create_connectivity_dict()
        
        # we make a copy as to not modify the original fully connected one
        connectivity_dict = copy.deepcopy(all_reaches_connectivity_dict)

        self.all_reaches_connectivity_dict = all_reaches_connectivity_dict

        # Identify ignored and upstream reaches
        ignored_reaches, upstream_of_ignored_reaches = self._find_ignored_reaches(connectivity_dict)

        # Set ignored reaches and remove from dictionary
        self._set_ignored_reaches(connectivity_dict, ignored_reaches, upstream_of_ignored_reaches)

        self.current_connectivity_dict = connectivity_dict
        return connectivity_dict
            
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
        for reach in self.reaches.values():
            if reach.strahler_number <= strahler_threshold:
                reach.ignore_reach()

    def assign_strahler_number_to_reaches(self, mode='fully_connected'):
        """
        Assign Strahler number to reaches in the system according to the connectivity dict.
        mode: 'fully_connected' (mode) -> Uses the fully connected network
              'current' -> Uses the current connectivity dict taking into account ignored reaches
        """

        reaches = self.reaches

        if mode == 'fully_connected':
            connectivity_dict = self.all_reaches_connectivity_dict
        elif mode == 'current':
            connectivity_dict = self.current_connectivity_dict
        else:
            warnings.warn("Invalid mode to assign strahler number -> Using 'fully_connected' by default")
            connectivity_dict = self.all_reaches_connectivity_dict
               

        root_id = self._find_root_reach(connectivity_dict)

        # optimizing the order of reaches processed by using DFS algorithm
        queue = dfs_iterative_postorder(connectivity_dict, root_id)
        # queue = list(connectivity_dict.keys())

        # Process the queue
        while queue:
            current_id = queue.pop(0) # take the first element
            current_reach = reaches[current_id]

            if current_id not in connectivity_dict:
                # reach is at the most upstream end
                current_reach.strahler_number = 1
                continue

            # get list of upstream reaches
            upstream_reaches_strahler = [reaches[id].strahler_number for id in connectivity_dict[current_id]]

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