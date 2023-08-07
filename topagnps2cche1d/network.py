from topagnps2cche1d.tools import dfs_iterative_postorder
import pandas as pd

class Node:
    def __init__(self, id, type=None, usid=None, us2id=None, dsid=None, computeid=None, x=None, y=None):
        self.id = id
        self.type = type
        self.usid = usid
        self.us2id = us2id
        self.dsid = dsid
        self.computeid = computeid
        self.x = x
        self.y = y

class Reach:
    def __init__(self, id, ignore=False, us_nd_id=None, ds_nd_id=None, receiving_reach_id=None):
        self.id = id
        self.ignore = ignore
        self.strahler_number = None
        self.us_nd_id = us_nd_id
        self.ds_nd_id = ds_nd_id
        self.receiving_reach_id = receiving_reach_id
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.id] = node

    def ignore_reach(self):
        self.ignore = True
    
    def include_reach(self):
        self.ignore = False

class Cell:
    def __init__(self, id, receiving_reach_id):
        self.id = id
        self.cell_type = None # Left, Right, Source
        self.receiving_reach_id = receiving_reach_id

class System:
    def __init__(self):
        self.reaches = {}
        self.connectivity_dict = {}

    def add_reach(self, reach):
        self.reaches[reach.id] = reach

    def assess_connectivity(self):
        
        connectivity_dict = {}

        for reach_id, reach in self.reaches.items():
            
            if reach.ignore:
                continue

            receiving_reach_id = reach.receiving_reach_id

            # Initialize the list for each channel if not already present
            if receiving_reach_id not in connectivity_dict:
                connectivity_dict[receiving_reach_id] = [reach_id]
            else:
                connectivity_dict[receiving_reach_id].append(reach_id)

        del connectivity_dict[None] # delete the "None" downtream reach of root

        # # Remove reaches that are ignored 
        # # (they are kept if they are upstream of a reach that is not ignored)
        # for id, reach in self.reaches.items():
        #     if reach.ignore:
        #         del connectivity_dict[id]

        self.connectivity_dict = connectivity_dict

    def find_root_reach(self):
        connectivity_dict = self.connectivity_dict

        all_upstream_reaches = []

        for upstream_reaches in connectivity_dict.values():
            all_upstream_reaches.extend(upstream_reaches)

        for reach_id in self.reaches.keys():
            if reach_id not in all_upstream_reaches:
                return reach_id
            
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
            reach.ignore = True

    def assign_strahler_number_to_reaches(self):

        reaches = self.reaches

        connectivity_dict = self.connectivity_dict
        
        root_id = self.find_root_reach()

        # optimizing the order of reaches processed by using DFS algorithm
        queue = dfs_iterative_postorder(connectivity_dict, root_id)

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