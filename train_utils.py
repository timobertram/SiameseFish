import chess
import torch
import os


piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

def get_engine():
    STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
    # make sure stockfish environment variable exists
    if STOCKFISH_ENV_VAR not in os.environ:
        os.environ['STOCKFISH_EXECUTABLE'] = "/home/fawler/tbertram/RBC/stockfish_14.1_linux_x64_ssse"
        

    # make sure there is actually a file
    stockfish_path = os.environ[STOCKFISH_ENV_VAR]
    if not os.path.exists(stockfish_path):
        raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

    # initialize the stockfish engine
    return chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)

def evaluate_board(board, time = 0.1):
    my_color = board.turn
    if board.attackers(my_color, board.king(not my_color)):
        return 1
    elif board.attackers(not my_color, board.king(my_color)):
        return 0
    else:
        try:
            score = get_engine().analyse(board,chess.engine.Limit(time= time))['score'].relative
            numerical_score = score.score()
            if numerical_score is None:
                mate = score.mate()
                if mate > 0:
                    return 1
                else:
                    return 0
            else:
                numerical_score /= 100
                new_win_chance = 1/(1+10**(-numerical_score/4))
            return new_win_chance
        except Exception as e:
            print('Something bad happened when choosing a move')
            print(f'{e} happened at board {board}')
            return None
