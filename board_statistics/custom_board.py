import chess
from typing import Union


class CustomBoard(chess.Board):
    """
    Class that inherits from chess.Board and extends the board by custom attributes.
    If the board is being copied with the inherited function .copy, then the custom attributes
    must be manually copied with copy_custom_attributes.
    """
    def __init__(self, *args):
        super().__init__()
        # we can pass an existing chess.Board instance as *args. The new board inherits everything from the existing one
        if type(args[0]) == chess.Board:
            self.__dict__ = args[0].__dict__.copy()

        # here we add our new custom attributes
        self.score: (int, int) = ()  # a tuple with 2 integer values. our score and enemy score
        self.last_e_to_square: Union[int, None] = None  # either the index of the last move or None in case of passing
        self.score_percentage: (int, int) = ()  # softmax over all board-scores

    def copy_custom_attributes(self, board):
        """
        Copies the custom attributes from a desired board to this board

        Args:
            board: (chess.board) Board from which the custom attributes should be copied

        Returns:

        """
        self.score = board.score
        self.last_e_to_square = board.last_e_to_square
        self.score_percentage = board.score_percentage
