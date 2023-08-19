import pandas as pd


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

    def get_nodes_us_ds_order(self):
        """
        Get the list of node ids in US -> DS direction [us_nd_id, ..., ds_nd_id]
        """
        nodes = self.nodes
        num_nodes_left = len(nodes)

        nodes_order = []

        current_node = nodes[self.us_nd_id]

        while num_nodes_left != 0:
            nodes_order.append(current_node.id)
            num_nodes_left -= 1

            if current_node.dsid not in nodes:
                break

            current_node = nodes[current_node.dsid]

        return nodes_order

    def resample_reach(self, **kwargs):
        """
        Resample reach either at a constant spacing or a given number of points (>=2)
        """

        if "distance" in kwargs:
            distance = kwargs["distance"]
        else:
            distance = None

        if "numpoints" in kwargs:
            numpoints = kwargs["numpoints"]
        else:
            numpoints = 30

    def get_x_y_node_arrays_us_ds_order(self):
        """
        Retrieve x and y node coordinates in US -> DS order
        """
        x = []
        y = []
        nodes_order = self.get_nodes_us_ds_order()
        nodes = self.nodes

        for node_id in nodes_order:
            x.append(nodes[node_id].x)
            y.append(nodes[node_id].y)

        return x, y

    def get_nodes_as_df(self):
        """
        Retrieve nodes information and put it in a DataFrame
        """
        nodes = self.nodes

        node_id = []
        node_type = []
        node_usid = []
        node_us2id = []
        node_dsid = []
        node_computeid = []
        node_x = []
        node_y = []
        node_row = []
        node_col = []

        for node in nodes.values():
            node_id.append(node.id)
            node_type.append(node.type)
            node_usid.append(node.usid)
            node_us2id.append(node.us2id)
            node_dsid.append(node.dsid)
            node_computeid.append(node.computeid)
            node_x.append(node.x)
            node_y.append(node.y)
            node_row.append(node.row)
            node_col.append(node.col)

        df = pd.DataFrame(
            {
                "ID": node_id,
                "COMPUTEID": node_computeid,
                "TYPE": node_type,
                "USID": node_usid,
                "US2ID": node_us2id,
                "DSID": node_dsid,
                "X": node_x,
                "Y": node_y,
                "ROW": node_row,
                "COL": node_col,
            }
        )

        df["Reach_ID"] = self.id
        df["CCHE1D_ID"] = self.cche1d_id
        df["Strahler_Number"] = self.strahler_number
        df = df.sort_values(by="COMPUTEID")

        return df
