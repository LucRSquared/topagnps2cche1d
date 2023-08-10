from topagnps2cche1d import Watershed


class Cell(Watershed):
    def __init__(self, id, type=None, receiving_reach_id=None):
        self.id = id
        self.type = type  # Can be 'left', 'right', or 'source'
        self.receiving_reach_id = receiving_reach_id
