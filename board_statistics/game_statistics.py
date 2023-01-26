from board_statistics.src import *
from board_statistics.square import Square
from collections import defaultdict
import numpy as np
import pandas as pd

class GameStatistics:
    def __init__(self, game):
        """
        Statistics class which collects and provides all relevant information about the game.

        Args:
            game: Multiple_Board game instance
        """
        self.evaluation_time_limit = 0.01
        self.game = game
        # we store the information for every square in a class instance
        self.squares = [Square(index=index) for index in range(64)]
        # the indices of our pieces (must be the same on all boards)
        self.piece_indices = []

        # does not work if statistics is created before handle_game_start
        # self.piece_indices = [index for index in list(range(64)) if self.game.board_dict.get_boards()[0].piece_at(index) and
        #                       self.game.board_dict.get_boards()[0].piece_at(index).color is self.game.color]

        self.non_piece_indices = []  # indices of squares that where our pieces are NOT located

        self.elimination_list = [0 for _ in range(64)]  # stores the minimum eliminations per square

        # we store the mean board evaluation after our move and after the enemy move and sensing
        self.mean_enemy_score_after_move = None
        self.mean_enemy_score_after_sensing = None
        # for each square - stores if piece on this square can king capture in 1 or 2 turns
        self.aggressive_square_list = [0 for _ in range(64)]
        self.use_siamese = True


    def zero_boards_check(self):
        """
        Checks if we lost all boards and raises an error if this is the case
        """
        if self.game.board_dict.size() == 0:
            raise ValueError("Cannot update statistics with no boards")

    def update_statistics(self):
        """
        Recomputes all relevant information about the game

        Returns: True if everything was updated

        """
        #self.zero_boards_check()
        #self.update_piece_indices()
        #self.update_board_evaluations()
        #self.update_board_score_percentage()
        self.delete_history()
        #self.update_squares()
        #self.update_aggressive_square_list()
        return True

    def update_squares(self):
        self.update_piece_indices()
        self.update_non_piece_indices()
        self.update_piece_list()
        self.update_min_eliminations()
        #self.update_to_square_evaluations()
        #self.update_to_square_probability()
        #self.update_king_capture_in()

    def update_board_evaluations(self):
        """
        Evaluates every board and stores the score for both players in the board attribute (as tuple of length 2).
        First entry: our score
        Second entry: enemy score

        """

        # time_limit = self.evaluation_time_limit
        time_limit = 1/self.game.board_dict.size()  # we set the time limit such that all boards are evaluated in 1 sec
        for board in self.game.board_dict.get_boards():
            board_score_self = get_board_evaluation(game=self.game, board=board, board_color=self.game.color,
                                                    time_limit=time_limit)
            board_score_opponent = - board_score_self
            board.score = (board_score_self, board_score_opponent)

    def update_board_score_percentage(self):
        board_score_list_self = [board.score[0]/100 for board in self.game.board_dict.get_boards()]
        board_score_list_enemy = [board.score[1]/100 for board in self.game.board_dict.get_boards()]
        board_score_distribution_self = softmax(board_score_list_self)
        board_score_distribution_enemy = softmax(board_score_list_enemy)
        for board, board_score_percentage_self, board_score_percentage_enemy in \
                zip(self.game.board_dict.get_boards(), board_score_distribution_self, board_score_distribution_enemy):
            board.score_percentage = (board_score_percentage_self, board_score_percentage_enemy)

    def get_mean_board_score(self):
        """
        Return mean score of all boards.
        Returns None if the boards have not been evaluated or if there are no boards.
        """
        board_eval_self = [board.score[0] if board.score else None for board in self.game.board_dict.get_boards()]
        board_eval_enemy = [board.score[1] if board.score else None for board in self.game.board_dict.get_boards()]

        if board_eval_self and None not in board_eval_self:  # if every board has an evaluation
            return np.mean(board_eval_self), np.mean(board_eval_enemy)

    def update_mean_enemy_score_after_move(self):
        """
        Stores the mean of all evaluations. Should only be called directly after we move.
        """
        mean_board_score = self.get_mean_board_score()
        if mean_board_score:
            self.mean_enemy_score_after_move = mean_board_score[1]

    def update_mean_enemy_score_after_sensing(self):
        """
        Stores the mean of all evaluations. Should only be called directly after sensing.
        """
        mean_board_score = self.get_mean_board_score()
        if mean_board_score:
            self.mean_enemy_score_after_sensing = self.get_mean_board_score()[1]

    def update_aggressive_square_list(self):
        """
        Updates aggressive_square_list which stores in how many moves a piece on each square can reach our king
        """
        # reset values
        self.aggressive_square_list = [0 for _ in range(64)]
        for square in self.squares:
            king_capture_in = square.king_capture_in
            if king_capture_in != -1:
                self.aggressive_square_list[square.index] = king_capture_in

    def update_sensing_strategy(self):
        """
        Updating sensing strategy after handling sensing results.
        If enemy plays aggressive, we increase panic. If the enemy plays normal, we sense best "normal" move.
        This is done by comparing board evaluations before and after player turns.
        If the enemy score increased or stayed the same, then the enemy plays normal.
        If the score decreased, then the enemy must plan an agressive move.
        """
        # Explanation:
        # we move. then we handle move result. then we make 1. measurement.
        # then enemy moves. then handle enemy move. then we sense. then we handle sense. then we make 2. measurement
        # then we compare the ratio of the two measurements
        if self.mean_enemy_score_after_sensing and self.mean_enemy_score_after_move:  # in beginning we have no scores
            improvement_factor = self.mean_enemy_score_after_sensing / self.mean_enemy_score_after_move
        else:
            improvement_factor = None
        # print("improvement_factor: ", improvement_factor)
        # print("mean score: ", self.mean_enemy_score_after_sensing)
        # TODO
        # change sensing based on these scores once the "king capture in 2" computation is fully functional

    def delete_history(self):
        """
        Removes the history of all boards. This may save some memory
        """
        for board in self.game.board_dict.get_boards():
            board.clear_stack()

    def update_piece_indices(self):
        """
        Updates self.piece_indices, and self.non_piece_indices
        """
        self.piece_indices = [index for index in list(range(64)) if self.game.board_dict.get_boards()[0].piece_at(index) and
                              self.game.board_dict.get_boards()[0].piece_at(index).color is self.game.color]

    def update_non_piece_indices(self):
        """
        Updates non_piece_indices which contains all indices of squares where there's no piece from our color
        """
        self.non_piece_indices = [index for index in list(range(64)) if index not in self.piece_indices]

    def update_piece_list(self):
        """
        Updates piece_dict of each square (occurrences of each piece per square)
        """
        for square in self.squares:  # resets dictionary
            for key in square.piece_dict.keys():
                square.piece_dict[key] = 0

        non_piece_square = [self.squares[index] for index in self.non_piece_indices]  # all relevant squares


        if self.use_siamese and self.game.timer.remaining() > 300:
            weighted_boards = self.game.reduce_boards(1000)
            if self.game.save_csv:
                df = pd.DataFrame(weighted_boards)
                df.to_csv(f'debugging/sense_distances_{len(self.game.siamese.board_list)+1}.csv', sep = ';', index = False)
            for square in non_piece_square:  # count and store occurrences
                for board,weight in weighted_boards:
                    # if weight < 0.5: break
                    piece = board.piece_at(square.index)
                    square.piece_dict[piece] += weight
        else:
            boards = self.game.board_dict.get_boards()
            for square in non_piece_square:  # count and store occurrences
                for board in boards:
                    piece = board.piece_at(square.index)
                    square.piece_dict[piece] += 1


    def update_min_eliminations(self):
        """
        Updates the lower bound of how many board we can eliminate when sensing a given square
        """
        for square in self.squares:  # reset the minimum eliminations
            square.min_eliminations = 0

        non_piece_square = [self.squares[index] for index in self.non_piece_indices]  # all relevant squares

        for square in non_piece_square:
            # number of occurrences of the most frequent piece on this square
            largest_occur = max(square.piece_dict.values())
            # this is the least amount of board we eliminate when sensing this square
            min_eliminations = sum(square.piece_dict.values()) - largest_occur
            square.min_eliminations = min_eliminations
            self.elimination_list[square.index] = min_eliminations  # stores all min_eliminations in list

    def update_to_square_evaluations(self):
        """
        Updates square.to_square_piece_dict, square.to_square_piece_dict_softmax, square.piece_distribution
        """
        for square in self.squares:  # reset dictionary
            square.to_square_piece_dict = defaultdict(list)
            square.to_square_piece_dict_softmax = defaultdict(list)
            square.piece_distribution = defaultdict(float)

        # update to_square_piece_dict
        for board in self.game.board_dict.get_boards():
            to_square_index = board.last_e_to_square  # index of the latest to_square on this board
            if to_square_index is not None:
                to_square_piece = board.piece_at(to_square_index)  # chess piece on this square
                to_square = self.squares[to_square_index]  # retrieve the corresponding square
                e_score = board.score[1]  # enemy score
                to_square.to_square_piece_dict[to_square_piece].append(e_score)  # add score to storage

        for square in self.squares:
            # stores all values of all keys in a list
            flattened_scores = [score/100 for piece_score_list in square.to_square_piece_dict.values()
                                for score in piece_score_list]

            # update to_square_piece_dict_softmax
            for key, piece_score_list in square.to_square_piece_dict.items():
                piece_score_list = [piece_score / 100 for piece_score in piece_score_list]  # centipawn to pawn score
                softmax_scores = custom_softmax(piece_score_list, flattened_scores)
                square.to_square_piece_dict_softmax[key] = softmax_scores

            # update piece_distribution
            for key, piece_score_list in square.to_square_piece_dict_softmax.items():
                square.piece_distribution[key] = np.sum(piece_score_list)

    def update_to_square_probability(self):
        """
        Updates square.to_square_probability, where each to_square is weighted with the board score as percentage
        """
        for square in self.squares:  # reset values
            square.to_square_probability = 0

        for board in self.game.board_dict.get_boards():
            if board.last_e_to_square:
                to_square_index = board.last_e_to_square
                square = self.squares[to_square_index]
                # add the softmax value of the board score to this square
                square.to_square_probability += board.score_percentage[1]  # enemy move -> enemy score

    def update_king_capture_in(self):
        """
        Update square.king_caputure_in to store in how many moves a piece on each square can capture our king
        """
        # reset values
        for square in self.squares:
            square.king_capture_in = -1

        # get information tuples for all boards
        danger_info_boards = Dangerous_squares(multiple_board_instance=self.game, time_limit=200).check_all([1, 6])
        # print(danger_info_boards)
        for info_tuples in danger_info_boards:  # iterate over all boards
            # iterate over all dangerous pieces of given board
            for atk_piece_index, to_square, capture_in, piece_type in info_tuples:
                # print("atk_piece_index:", atk_piece_index)
                cur_square = self.squares[atk_piece_index]
                stored_threat = cur_square.king_capture_in
                # if king can be captured in 1 and and 2, we store 1
                if stored_threat != -1 and stored_threat < capture_in:
                    continue
                else:  # if we store -1 or 2, we can safely store new value
                    cur_square.king_capture_in = capture_in
