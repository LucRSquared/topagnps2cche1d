from tools import dfs_iterative_postorder

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
    def __init__(self, id, ignore=False, strahler_order=None, us_nd_id=None, ds_nd_id=None, receiving_reach_id=None):
        self.id = id
        self.ignore = ignore
        self.strahler_order = strahler_order
        self.us_nd_id = us_nd_id
        self.ds_nd_id = ds_nd_id
        self.receiving_reach_id = receiving_reach_id
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    def ignore_reach(self):
        self.ignore = True
    
    def include_reach(self):
        self.ignore = False

class System:
    def __init__(self):
        self.reaches = []
        self.connectivity_dict = {}

    def add_reach(self, reach):
        self.reaches.append(reach)

    def assess_connectivity(self):
        
        connectivity_dict = {}

        for reach in self.reaches:
            
            if reach.ignore:
                continue

            reach_id = reach.id
            receiving_reach_id = reach.receiving_reach_id

            # Initialize the list for each channel if not already present
            if receiving_reach_id not in connectivity_dict:
                connectivity_dict[receiving_reach_id] = [reach_id]
            else:
                connectivity_dict[receiving_reach_id].append(reach_id)

        self.connectivity_dict = connectivity_dict

    def find_root_reach(self):
        connectivity_dict = self.connectivity_dict

        for reach_test_root in self.reaches:
            for upstream_reaches in connectivity_dict.values():
                if reach_test_root.id not in upstream_reaches:
                    return reach_test_root.id


    def assign_strahler_number_to_reaches(self):
        # def compute_

        connectivity_dict = self.connectivity_dict

        for reach in self.reaches:
            if reach.id not in connectivity_dict:
                reach.strahler_number = 1
