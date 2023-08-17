from topagnps2cche1d.tools import rowcol2latlon_esri_asc
import numpy as np


class Node:
    def __init__(
        self,
        id,
        type=None,
        usid=None,
        us2id=None,
        dsid=None,
        computeid=None,
        x=None,
        y=None,
        row=None,
        col=None,
    ):
        self.id = id
        self.usid = usid
        self.dsid = dsid
        self.us2id = us2id

        self.type = type

        self.computeid = computeid

        self.x = x
        self.y = y
        self.row = row
        self.col = col

        self.inflow_cell_source = None
        self.inflow_reaches_source = []

    def __str__(self):
        out_str = [
            "-----------------------------",
            f"Node ID         : {self.id}",
            f"TYPE            : {self.type}",
            f"USID            : {self.usid}",
            f"DSID            : {self.dsid}",
            f"US2ID           : {self.us2id}",
            f"COMPUTEID       : {self.computeid}",
            f"Cell Source     : {self.inflow_cell_source}",
            f"Reach Sources   : {self.inflow_reaches_source}",
            f"(x,y)           : ({self.x}, {self.y})",
            f"(row,col)       : ({self.row}, {self.col})",
        ]

        return "\n".join(out_str)

    def set_node_type(self, type):
        self.type = type

    def change_node_ids_dict(self, old_new_dict):
        """
        Given a dictionary with {old_id: new_id} structure, applies the id change to the node (except computeid)
        """
        # Add fixed dictionary points
        old_new_dict[None] = None
        old_new_dict[-1] = -1

        self.id = old_new_dict[self.id]
        self.usid = old_new_dict[self.usid]
        self.us2id = old_new_dict[self.us2id]
        self.dsid = old_new_dict[self.dsid]

    def compute_XY_coordinates(self, geomatrix, oneindexed=False):
        """
        If the node has ROW/COL information, this function computes the XY coordinates
        The provided row col NEED to be in 0-index. If oneindexed is provided (i.e. input assumes that the first row is row = 1
        then an adjustment needs to be done
        """
        self.y, self.x = rowcol2latlon_esri_asc(
            geomatrix, self.row, self.col, oneindexed=oneindexed
        )

    def distance_from(self, other, measure="euclidean"):
        if measure.lower() in ["euclidean", "l2"]:
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        elif measure.lower() in ["manhattan", "l1"]:
            return np.abs(self.x - other.x) + np.abs(self.y - other.y)
