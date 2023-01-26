from collections import defaultdict


class Square:
    def __init__(self, index):
        """
        Stores the information of a single chess-board-square
        Args:
            index (int): Index of the square (where it is located on the board)
        """
        self.index = index
        self.piece_dict = defaultdict(int)  # stores how many times a piece occurs on this square
        self.min_eliminations = 0

        # this is a dictionary with every chess piece as key
        # if the latest to_square on a board is this square, we add an entry to the list of
        # the corresponding piece that moved to this square
        #  these entries are the evaluations of individual boards from the opponents point of view
        # if we have 7 boards where the latest move was to this square, the dict should then look something like this:
        # "r": [200, 500, -200], "p": [20, 23, 300], "q": [-1000]
        self.to_square_piece_dict = defaultdict(list)

        # same as to_square_piece_dict but we apply softmax such that the sum of ALL lists sum up to 1
        self.to_square_piece_dict_softmax = defaultdict(list)

        # same as to_square_piece_dict_softmax but the list is replaced with the sum of the list
        # this should reflect the probability for each piece to be true piece that was used for the lastest move
        self.piece_distribution = defaultdict(float)

        # stores how likely it is that the latest move was a move to this square
        self.to_square_probability = 0

        # stores how in how many moves any piece on this square can capture our king
        self.king_capture_in = -1
