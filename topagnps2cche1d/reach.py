class Reach:
    def __init__(self, id, receiving_reach_id=None, slope=None, ignore=False):
        self.id = id

        self.ignore = ignore

        self.receiving_reach_id = receiving_reach_id
        self.slope = slope

        self.cche1d_id = None

        self.strahler_number = None
        self.us_nd_id = None
        self.ds_nd_id = None

        self.nodes = {}

    def __str__(self):
        out_str = [
            "-----------------------------",
            f"Reach            : {self.id}",
            f"CCHE1D Ch. ID    : {self.cche1d_id}",
            f"Receiving Reach  : {self.receiving_reach_id}",
            f"Slope            : {self.slope}",
            f"Strahler Number  : {self.strahler_number}",
            f"Ignore?          : {self.ignore}",
            f"(US/DS) Nodes    : ({self.us_nd_id}, {self.ds_nd_id})",
        ]

        return "\n".join(out_str)

    def add_node(self, node):
        self.nodes[node.id] = node

    def ignore_reach(self):
        """
        Set a reach to be ignored in subsequent computation
        """
        self.ignore = True

    def include_reach(self):
        """
        Set a reach to be kept in subsequent computation
        """
        self.ignore = False

    def flip_reach_us_ds_order(self):
        self.us_nd_id, self.ds_nd_id = self.ds_nd_id, self.us_nd_id
        for node in self.nodes.values():
            node.usid, node.dsid = node.dsid, node.usid

    def compute_XY_coordinates_of_reach_nodes(self, geomatrix, oneindexed=False):
        """
        Compute the XY coordinates of all the nodes in the reach
        - oneindexed=False assumes that the row col coordinates are provided in the 0-starting format
        """
        nodes = self.nodes

        for node in nodes.values():
            node.compute_XY_coordinates(geomatrix, oneindexed=oneindexed)

    def change_node_ids_dict(self, old_new_dict):
        """
        Given a dictionary with {old_id: new_id} structure, applies the id change to the nodes
        """
        # Add fixed dictionary points
        old_new_dict[None] = None
        old_new_dict[-1] = -1

        self.us_nd_id = old_new_dict[self.us_nd_id]
        self.ds_nd_id = old_new_dict[self.ds_nd_id]

        # Remap nodes dict too:
        self.nodes = {old_new_dict[k]: v for k, v in self.nodes.items()}
