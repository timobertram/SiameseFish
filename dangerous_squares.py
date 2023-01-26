# imports
from reconchess import *
import chess.svg
import chess.engine

import time
import copy
import warnings


class Dangerous_squares:
    def __init__(self, multiple_board_instance, color: Color = None, time_limit: int = 5, top_n_boards: int = 50):
        """
        This class checks all given 'top n' boards in MultipleBoard for dangerous positions of given piece type.
        The evaluation is stopped and the result returned when the 'time_limit' clock is reached.

        Example:   danger_pos = Dangerous_squares(self.multiple_board_instance, time_limit=200).check_all([1, 6])

        @param multiple_board_instance: the MultipleBoard class
        @param color: color of the player who is attacked
        @param time_limit: time limit for the computation
        @param top_n_boards: the number of most interesting boards the function will look at
        """
        self.multiple_board_instance = multiple_board_instance
        if color is not None:
            self.color = color
        else:
            self.color = self.multiple_board_instance.color
        self.board_dict = self.multiple_board_instance.board_dict
        self.debug = self.multiple_board_instance.debug

        self.time_limit = time_limit
        self.starting_time = None
        self.time_offset = 0.1

        if top_n_boards == 0:
            self.board_limit = False
        else:
            self.board_limit = True
        self.top_n_boards = top_n_boards

    def check_all(self, exclude=None) -> list:
        """
        Checks all positions of all pieces excluding given ones.
        Aborts if the time limit is reached.
        @param exclude: pieces one wants to exclude from the check
        @return: list of integers with all dangerous positions
        """
        if exclude is None:
            exclude = list()
        self.starting_time = time.time()
        danger_squares = None
        pieces = [x for x in range(1, 7) if x not in exclude]
        for piece in pieces:
            res_pos = self.check(piece)
            if res_pos is None:
                continue
            elif danger_squares is None:
                danger_squares = res_pos
            else:
                for i in range(len(danger_squares)):
                    danger_squares[i].extend(res_pos[i])
        return danger_squares

    def check_board(self, board: chess.Board, exclude=None) -> list:
        """
        Checks all positions of all pieces excluding given ones.
        Aborts if the time limit is reached.
        @param board: board to be checked
        @param exclude: pieces one wants to exclude from the check
        @return: list of integers with all dangerous positions
        """
        if exclude is None:
            exclude = list()
        self.starting_time = time.time()
        danger_squares = None
        pieces = [x for x in range(1, 7) if x not in exclude]
        for piece in pieces:
            res_pos = self.check(piece, board)
            if res_pos is None:
                continue
            elif danger_squares is None:
                danger_squares = res_pos
            else:
                for i in range(len(danger_squares)):
                    danger_squares[i].extend(res_pos[i])
        return danger_squares

    def check(self, piece: int, board: chess.Board = None, time_limit: int = None) -> list:
        """
        Checks all boards for dangerous positions of the given piece.
        @param board: board to be checked; None to check all
        @param piece: current piece type the function should handle
        @param time_limit: time limit if the function is called individually
        @return: list of int's containing possibly dangerous positions
        """
        # if the function is called individually -> time is set -> attributes have to be updated
        if time_limit is not None:
            self.starting_time = time.time()
            self.time_limit = time_limit

        if board is None:
            if self.board_limit:
                selected_boards = copy.deepcopy(sorted(self.board_dict.get_boards(),
                                                       key=lambda board_: board_.score)[:self.top_n_boards])
            else:
                selected_boards = copy.deepcopy(self.board_dict.get_boards())
            if piece == chess.KNIGHT:
                danger_positions = self._check_knight_positions(selected_boards)
            elif piece == chess.PAWN:
                # TODO: Add dangerous positions with pawns
                warnings.warn("Checking for dangerous pawns is currently not implemented!")
                danger_positions = None
            elif piece == chess.KING:
                warnings.warn("Checking for dangerous king is currently not implemented!")
                danger_positions = None
            else:
                danger_positions = self._check_linear_pieces(piece, selected_boards)
        else:
            if piece == chess.KNIGHT:
                danger_positions = self._check_knight_positions([board])
            elif piece == chess.PAWN:
                # TODO: Add dangerous positions with pawns
                warnings.warn("Checking for dangerous pawns is currently not implemented!")
                danger_positions = None
            elif piece == chess.KING:
                warnings.warn("Checking for dangerous king is currently not implemented!")
                danger_positions = None
            else:
                danger_positions = self._check_linear_pieces(piece, [board])

        return danger_positions

    def _check_knight_positions(self, selected_boards: list):
        """
        check all boards for dangerous knights positions
        @param selected_boards: boards to be checked
        @return: list with all potential dangerous positions of knights
        """
        piece = chess.KNIGHT
        res_list = list()
        for board in selected_boards:
            # check if time limit is reached
            if time.time() - self.starting_time + self.time_offset >= self.time_limit:
                if self.debug:
                    print('Time expired in DangerousSquares.check(), returning results.')
                    break
            check_pos = list()
            board_res_list = list()
            # check possible moves of 'piece' which can reach the king in 2 moves
            enemy_pieces_list = board.pieces(piece, not self.color).tolist()
            enemy_pieces_pos = [i for i, x in enumerate(enemy_pieces_list) if x]
            try:
                own_king = board.pieces(chess.KING, self.color).tolist().index(True)
            except ValueError as val_err:
                # found board where we lost the game (no own king)
                return []
            piece_board = chess.Board(fen=None)
            piece_board.set_piece_at(own_king, chess.Piece(piece, self.color))
            piece_moves = list(piece_board.legal_moves)
            pot_danger_pos = [piece_move.to_square for piece_move in piece_moves]
            check_pos = [pos for pos in pot_danger_pos if pos in enemy_pieces_pos]
            # create pot. dangerous 'piece' positions which are 2 turn away
            piece_moves_2 = list()
            for pos in pot_danger_pos:
                piece_board_2 = chess.Board(fen=None)
                piece_board_2.set_piece_at(pos, chess.Piece(piece, self.color))
                piece_moves_2.extend(piece_board_2.legal_moves)
            pot_danger_pos_2 = sorted(list(set([piece_move.to_square for piece_move in piece_moves_2])))
            try:
                pot_danger_pos_2.remove(own_king)
            except Exception as e:
                # print("No own king in list - board: {}".format(board.board_fen()))
                pass
            pot_danger_pos_2 = [e for e in pot_danger_pos_2 if e not in pot_danger_pos]
            danger_pos_2 = [pos for pos in pot_danger_pos_2 if pos in enemy_pieces_pos]

            ret_check_pos = list()
            for d_pos in danger_pos_2:
                piece_board_2 = chess.Board(fen=None)
                piece_board_2.set_piece_at(d_pos, chess.Piece(piece, self.color))
                legal_moves_backwards = piece_board_2.legal_moves
                ret_check_pos.extend([(d_pos, k_move.to_square) for k_move in legal_moves_backwards
                                      if k_move.to_square in pot_danger_pos])
            if len(check_pos) > 0:
                check_pos = list(dict.fromkeys(check_pos))
                for pos in check_pos:
                    board_res_list.append((pos, pos, 1, piece))

            ret_check_pos = list(dict.fromkeys(ret_check_pos))
            for enemy_pos, ret_pos in ret_check_pos:
                board_res_list.append((enemy_pos, ret_pos, 2, piece))

            res_list.append(board_res_list)
        return res_list

    def _check_linear_pieces(self, piece: int, selected_boards: list):
        res_list = list()
        for board in selected_boards:
            # check if time limit is reached
            if time.time() - self.starting_time + self.time_offset >= self.time_limit:
                if self.debug:
                    print('Time expired in DangerousSquares.check(), returning results.')
                    break
            board_res_list = list()
            enemy_pieces_list = board.pieces(piece, not self.color).tolist()
            enemy_pieces_pos = [i for i, x in enumerate(enemy_pieces_list) if x]
            try:
                own_king = board.pieces(chess.KING, self.color).tolist().index(True)
            except ValueError as val_err:
                # found board where we lost the game (no own king)
                continue

            if piece == chess.ROOK:
                # get the x & y lines according to the king position
                x_line, y_line = self._get_xy_lines(own_king, board)
                king_tiles = [*x_line, *y_line]
                # remove duplicates
                king_tiles = list(dict.fromkeys(king_tiles))
                for enemy_pos in enemy_pieces_pos:
                    x_line_rook, y_line_rook = self._get_xy_lines(enemy_pos, board)
                    rook_tiles = [*x_line_rook, *y_line_rook]
                    # remove duplicates
                    rook_tiles = list(dict.fromkeys(rook_tiles))
                    for king_tile in king_tiles:
                        if king_tile in rook_tiles:
                            if own_king in rook_tiles:
                                num_moves = 1
                            else:
                                num_moves = 2
                            board_res_list.append((enemy_pos, king_tile, num_moves, piece))
            elif piece == chess.BISHOP:
                # get the diagonal lines according to the king position
                pos_diag, neg_diag = self._get_diagonals(own_king, board)
                king_tiles = [*pos_diag, *neg_diag]
                # remove duplicates
                king_tiles = list(dict.fromkeys(king_tiles))
                for enemy_pos in enemy_pieces_pos:
                    pos_diag_bishop, neg_diag_bishop = self._get_diagonals(enemy_pos, board)
                    bishop_tiles = [*pos_diag_bishop, *neg_diag_bishop]
                    # remove duplicates
                    bishop_tiles = list(dict.fromkeys(bishop_tiles))
                    for king_tile in king_tiles:
                        if king_tile in bishop_tiles:
                            if own_king in bishop_tiles:
                                num_moves = 1
                            else:
                                num_moves = 2
                            board_res_list.append((enemy_pos, king_tile, num_moves, piece))
            elif piece == chess.QUEEN:
                # get the x & y lines according to the king position
                x_line, y_line = self._get_xy_lines(own_king, board)
                # get the diagonal lines according to the king position
                pos_diag, neg_diag = self._get_diagonals(own_king, board)
                king_tiles = [*x_line, *y_line, *pos_diag, *neg_diag]
                # remove duplicates
                king_tiles = list(dict.fromkeys(king_tiles))
                for enemy_pos in enemy_pieces_pos:
                    pos_diag_queen, neg_diag_queen = self._get_diagonals(enemy_pos, board)
                    x_line_queen, y_line_queen = self._get_xy_lines(enemy_pos, board)
                    queen_tiles = [*pos_diag_queen, *neg_diag_queen, *x_line_queen, *y_line_queen]
                    # remove duplicates
                    queen_tiles = list(dict.fromkeys(queen_tiles))
                    for king_tile in king_tiles:
                        if king_tile in queen_tiles:
                            if own_king in queen_tiles:
                                num_moves = 1
                            else:
                                num_moves = 2
                            board_res_list.append((enemy_pos, king_tile, num_moves, piece))
            else:
                raise ValueError(f"Piece {piece} cannot be handled in _check_linear_pieces()")

            res_list.append(board_res_list)

        return res_list

    def _get_xy_lines(self, piece: int, board: chess.Board):
        # get the row & col the king is in
        king_row = int(piece / 8)
        king_col = int(piece % 8)
        # compute x_line & y_line
        x_line_full = [i for i in range((king_row * 8), ((king_row * 8) + 8))]
        y_line_full = [i for i in range(king_col, (56 + king_col + 1), 8)]
        # get the indices of the current piece
        piece_x_index = x_line_full.index(piece)
        piece_y_index = y_line_full.index(piece)
        # compute the real lines
        x_line = self._cut_lines(piece_x_index, x_line_full, board)
        y_line = self._cut_lines(piece_y_index, y_line_full, board)

        return x_line, y_line

    def _get_diagonals(self, piece: int, board: chess.Board):
        pos_diag_full, neg_diag_full = [], []
        # compute pos_diag
        runner = piece
        while runner >= 8 and (runner % 8) != 0:
            runner -= 9
        pos_diag_full.append(runner)
        while runner not in [7, 15, 23, 31, 39, 47, 55, 56, 57, 58, 59, 60, 61, 62, 63]:
            runner += 9
            pos_diag_full.append(runner)
        # compute neg_diag
        runner = piece
        while runner % 8 != 0 and runner < 55:
            runner += 7
        neg_diag_full.append(runner)
        while runner not in [0, 1, 2, 3, 4, 5, 6, 7, 15, 23, 31, 39, 47, 55, 63]:
            runner -= 7
            neg_diag_full.append(runner)

        # get the indices of the piece in the diagonals
        piece_neg_index = neg_diag_full.index(piece)
        piece_pos_index = pos_diag_full.index(piece)
        # compute the real diagonals
        pos_diag = self._cut_lines(piece_pos_index, pos_diag_full, board)
        neg_diag = self._cut_lines(piece_neg_index, neg_diag_full, board)

        return pos_diag, neg_diag

    def _cut_lines(self, piece_index: int, line_full: list, board: chess.Board):
        # negative direction
        neg_cut = 0
        for i in range(piece_index - 1, -1, -1):
            piece_at_pos = board.piece_at(line_full[i])
            if piece_at_pos is not None:
                if piece_at_pos.color == self.color:
                    neg_cut = i
                else:
                    neg_cut = i + 1
                break
        # positive direction
        pos_cut = 8
        for i in range(piece_index + 1, len(line_full)):
            piece_at_pos = board.piece_at(line_full[i])
            if piece_at_pos is not None:
                if piece_at_pos.color == self.color:
                    pos_cut = i + 1
                else:
                    pos_cut = i
                break
        # cut the full line
        line = line_full[neg_cut:pos_cut]
        return line
