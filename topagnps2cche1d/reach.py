class Reach:
    def __init__(self, id, ignore=False, receiving_reach_id=None, slope=None):
        self.id = id
        self.ignore = ignore
        self.receiving_reach_id = receiving_reach_id
        self.slope = slope

        self.strahler_number = None
        self.us_nd_id = None
        self.ds_nd_id = None
    
        self.nodes = {}

    def __str__(self):
        out_str = [
            "-----------------------------",
            f"Reach           : {self.id}",
            f"Receiving Reach : {self.receiving_reach_id}",
            f"Slope           : {self.slope}",
            f"Strahler Number : {self.strahler_number}",
            f"Ignore?         : {self.ignore}"
        ]

        return '\n'.join(out_str)

    def add_node(self, node):
        self.nodes[node.id] = node

    def ignore_reach(self):
        self.ignore = True
    
    def include_reach(self):
        self.ignore = False