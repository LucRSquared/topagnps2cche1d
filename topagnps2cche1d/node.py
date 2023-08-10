from topagnps2cche1d.reach import Reach


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
