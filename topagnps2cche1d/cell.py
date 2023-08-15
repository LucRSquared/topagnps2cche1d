# from topagnps2cche1d.watershed import Watershed


class Cell:
    def __init__(self, id, type=None, area=None, receiving_reach_id=None):
        self.id = id

        self.area = area

        self.receiving_reach_id = receiving_reach_id
        self.receiving_node_id = None

        if type is None:
            self._determine_type_based_on_topagnps_id()
        else:
            self.type = type  # Can be 'left', 'right', or 'source'

    def _determine_type_based_on_topagnps_id(self):
        """
        This function determines based on the last digit of the cell_id. The convenction is as follows:
        - When looking in the downstream direction of a reach:
            - A source cell (if it exists) will finish with the digit 1
            - A right cell will finish with the digit 2
            - A left cell will finish with the digit 3
        """
        last_digit = self.id % 10

        if last_digit == 1:
            self.type = "source"
        elif last_digit == 2:
            self.type = "right"
        elif last_digit == 3:
            self.type = "left"
        else:
            raise Exception(f"Non recognized cell_id (last-digit) type: {last_digit}")

    def set_receiving_node(self, nd_id):
        self.receiving_node_id = nd_id
        # if self.type in ['left', 'right']:
        #     self.receiving_node_id = watershed.reaches[self.receiving_reach_id].ds_nd_id
        # elif self.type == 'source':
        #     self.receiving_node_id = watershed.reaches[self.receiving_reach_id].us_nd_id
