import numpy as np
from reconchess import *
import scipy.signal


def get_board_evaluation(game, board, board_color, time_limit=0.1):
    """Evaluates a board for a given player using stockfish

    Parameters:
    game (class): Multiple_Board game instance
    board (class): board that is to be evaluated
    board_color (bool): color of the player

    Returns:
    score_value (int): Score of the board according to Stockfish

   """

    # we create a new board from the current board to delete the history.
    # this allows for evaluations of boards with passing moves
    fen = board.board_fen()
    # since new boards always have the turn set to white, we must manually update the turn
    turn = board.turn
    fen_board = chess.Board(fen)
    fen_board.turn = turn

    # if we can capture the king, we cannot analyse this board as normal chess does not allow this
    e_king_square = fen_board.king(not fen_board.turn)  # get enemy king
    e_king_attackers = fen_board.attackers(fen_board.turn, e_king_square)
    if e_king_attackers:  # if we can take the king we define the evaluation
        score_value = 10000  # boards that result in an instant win are highly valuable
        if game.eval_debug:
            if fen_board.turn:  # True is WHITE, False is BLACK
                player_name = "WHITE"
            else:
                player_name = "BLACK"
            print(f"{player_name} can take the king on this board. Score: 10000 for {player_name}")
    else:  # if the king cannot be captured, we analyse the board
        results = game.engine.analyse(fen_board, chess.engine.Limit(time=time_limit))  # evaluate board

        score = results["score"]
        if game.eval_debug:
            print(score)

        score_string = str(score.relative)
        if score_string[0] == "#":  # mate was detected
            if int(score_string[1:]) > 0:  # we can mate
                score_value = 1000  # we attribute 10 pawns to a mate
            else:
                score_value = -1000  # we attribute 10 pawns to the enemy for enemy mate
        else:  # no mate
            score_value = int(score_string)

    # we flip the score if we want the evaluation for the opposing player
    if board_color != fen_board.turn:
        return - score_value
    return score_value


def to_board_matrix(list_values):
    """
    Converts list of shape 64,1 to np.array of shape 8,8
    Args:
        list_values: list that is to be reshaped

    Returns: reshaped np.array
    """
    return np.array(list_values).reshape(8, 8)


def convolution_sum(board_matrix):
    """
    Convolutes over a 8,8 matrix and computes the sum of a 3x3 kernel

    Args:
        board_matrix: 8,8 np.array from which the sum is computed

    Returns:
        np.array of shape 8,8 that contains all the 3x3 sums of the convolution
    """
    kernel = np.ones(shape=(3, 3))
    sum_array = scipy.signal.convolve2d(board_matrix, kernel, mode="same", boundary="fill")
    return sum_array


def softmax(array):
    """
    Applies softmax function to the input
    """
    array -= np.max(array)  # avoids overflow error
    array = np.exp(array)
    array /= np.sum(array)
    return array


def custom_softmax(array_small, array_large):
    """
    Same as regular softmax, but we manually specify the array that we divide by.
    (This way we can apply softmax to multiple arrays at once and the sum of all entries of all arrays sum up to 1)
    """
    array_small -= np.max(array_large)  # avoids overflow error
    array_large -= np.max(array_large)  # avoids overflow error
    array_small = np.exp(array_small)
    array_large = np.exp(array_large)
    array_small /= np.sum(array_large)
    return array_small
