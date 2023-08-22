# from topagnps2cche1d.watershed import Watershed


class Cell:
    def __init__(self, id, type=None, area=None, receiving_reach_id=None):
        self.id = id

        self.area = area

        self.receiving_reach_id = receiving_reach_id

        if type is None:
            self._determine_type_based_on_topagnps_id()
        else:
            self.type = type  # Can be 'left', 'right', or 'source'

    def __str__(self) -> str:
        out_str = [
            "-----------------------------",
            f"Cell ID            : {self.id}",
            f"Area               : {self.area}",
            f"Receiving Reach ID : {self.receiving_reach_id}",
            f"Type               : {self.type}"
        ]

        return "\n".join(out_str)

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
