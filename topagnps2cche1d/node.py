from topagnps2cche1d.reach import Reach
import numpy as np


class Node(Reach):
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
        self.type = type
        self.usid = usid
        self.us2id = us2id
        self.dsid = dsid
        self.computeid = computeid
        self.x = x
        self.y = y
        self.row = row
        self.col = col

    def distance_from(self, other, measure="euclidean"):
        if measure.lower() in ["euclidean", "l2"]:
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        elif measure.lower() in ["manhattan", "l1"]:
            return np.abs(self.x - other.x) + np.abs(self.y - other.y)
