from models import Evaluation_Network, Supervised_Dataset, replace_output_block
import torch
import chess
import os

piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

index_to_piece = {v:k for k,v in piece_to_index.items()}

def last_move_to_capture_square(move):
    return(int_to_row_column(move.to_square))


def int_to_row_column(location):
    return 7-location//8,location%8

def row_column_to_int(row,column):
    return (7-row)*8+column

def board_from_fen(board):
    tensor_board = torch.zeros(12,8,8)
    board = chess.Board(board)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
    return tensor_board

def load_network(path = 'networks/stockfish/tuned.pt'):
    network = Evaluation_Network(stockfish = False,num_first_blocks=0,num_second_blocks=40,first_block_size= 256, second_block_size=256)
    try:
        network.load_state_dict(torch.load(path))
    except RuntimeError:
        replace_output_block(network)
        network.load_state_dict(torch.load(path))
    network.eval()
    return network

def stockfish_eval(fen):
    board = chess.Board(fen)
    STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
    
    if STOCKFISH_ENV_VAR not in os.environ:
        os.environ['STOCKFISH_EXECUTABLE'] = "/home/fawler/tbertram/RBC/stockfish_14.1_linux_x64_ssse"
    stockfish_path = os.environ[STOCKFISH_ENV_VAR]
    if not os.path.exists(stockfish_path):
        raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

    # initialize the stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
    eval = engine.analyse(board,chess.engine.Limit(time=0.01))['score'].white()
    if eval.mate():
        if eval.mate() > 0:
            return 1
        else:
            return 0
    else:
        numerical_score = eval.score()/ 100
        win_chance = 1/(1+10**(-numerical_score/4))
        return win_chance


    