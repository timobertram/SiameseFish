import chess
import train_utils
import utils
import chess.engine
from argparse import ArgumentParser
import torch 
import os
import time
from torch import sigmoid
from scipy.stats import pearsonr   
from models import Supervised_Dataset, collate_fn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def eval_from_fen(fen, last_move = None, network = None):
    turn = chess.Board(fen).turn
    if turn:
        turn_tensor = torch.ones((1,8,8))
    else:
        turn_tensor = torch.zeros((1,8,8))
    capture_tensor = torch.zeros((1,8,8))
    if last_move is not None:
        print(UserWarning('Last move is not none, this will change the evaluation!'))
        row,col = utils.last_move_to_capture_square(last_move)
        capture_tensor[0,row,col] = 1
    board = utils.board_from_fen(fen)
    input = torch.cat((board,capture_tensor,turn_tensor),dim = 0)
    input = input.unsqueeze(dim=0)
    if network:
        model = utils.load_network(network)
    else:
        model = utils.load_network('networks/stockfish/tuned_v1.pt')
    eval = sigmoid(model(input))
    return eval.item()


def fen_from_input(input):
    input = input.view(-1,8,8)
    board = chess.Board()
    board.clear()
    for layer in range(input.shape[0]-1):
        pieces = (input[layer] == 1).nonzero(as_tuple=True)
        for p in range(pieces[0].size(0)):
            rank = chess.RANK_NAMES[7-pieces[0][p]]
            file = chess.FILE_NAMES[pieces[1][p]]
            square = chess.parse_square(file+rank)
            board.set_piece_at(square,chess.Piece.from_symbol(utils.index_to_piece[layer]))
    board.turn = True if input[-1,0,0] == 1 else False
    return board.fen()


def fixed_test_cases(network = None):
    print(f" Knight Mate in 1 for white: {eval_from_fen('rnbqkbnr/pppppppp/5N2/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/5N2/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1')}")
    print(f" Knight Mate in 1 for white (without capture flag) but white to play: {eval_from_fen('rnbqkb1r/ppp1pppp/3N1n2/3p4/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 3', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkb1r/ppp1pppp/3N1n2/3p4/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 3')}")
    print(f" Knight Mate in 1 for white (without capture flag) but black to play: {eval_from_fen('rnbqkb1r/ppp1pppp/3N1n2/3p4/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkb1r/ppp1pppp/3N1n2/3p4/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3')}")
    print(f" Knight Mate in 1 for white (with capture flag) but black to play: {eval_from_fen('rnbqkb1r/ppp1pppp/3N1n2/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3',chess.Move.from_uci('b5d6'), network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkb1r/ppp1pppp/3N1n2/8/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 3')}")
    print(f" Knight Mate in 1 for black: {eval_from_fen('r1bqkbnr/pppppppp/8/8/8/2n5/PPPPPPPP/RNBQKBNR b KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('r1bqkbnr/pppppppp/8/8/8/2n5/PPPPPPPP/RNBQKBNR b KQkq - 0 1')}")
    print(f" Bishop Mate in 1 for white: {eval_from_fen('rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/ppp2ppp/8/1B1pp3/4P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1')}")
    print(f" Almost Bishop Mate in 1 for white: {eval_from_fen('rnbqkb1r/pppp1ppp/5n2/1B2p3/4P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkb1r/pppp1ppp/5n2/1B2p3/4P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1')}")
    print(f" Start for white: {eval_from_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')}")
    print(f" 1.D4 : {eval_from_fen('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1')}")
    print(f" 1.D4 Pass: {eval_from_fen('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1')}")
    print(f" 1.E4: {eval_from_fen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')}")
    print(f" 1.Pass: {eval_from_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1', network = network)}")
    print(f"Stockfish {utils.stockfish_eval('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1')}")


def juffi_test_cases_notake(network = None):
    print(f" 2. Qa4 b5 3.Qxb5: {eval_from_fen('rnbqkbnr/p1p1pppp/3p4/1Q6/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")
    print(f" 2. Qa4 c6 3.Qxc6: {eval_from_fen('rnbqkbnr/pp2pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")
    print(f" 2. Qa4 Nc6 3.Qxc6: {eval_from_fen('r1bqkbnr/ppp1pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")
    print(f" 2. Qa4 Nd7 3.Qxd7: {eval_from_fen('r1bqkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")
    print(f" 2. Qa4 Bd7 3.Qxd7: {eval_from_fen('rn1qkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")
    print(f" 2. Qa4 Qd7 3.Qxd7: {eval_from_fen('rnb1kbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2', network = network)}")

def juffi_test_cases(network = None):
    print(f" 2. Qa4 b5 3.Qxb5: {eval_from_fen('rnbqkbnr/p1p1pppp/3p4/1Q6/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4b5'), network = network)}, Stockfish {utils.stockfish_eval('rnbqkbnr/p1p1pppp/3p4/1Q6/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")
    print(f" 2. Qa4 c6 3.Qxc6: {eval_from_fen('rnbqkbnr/pp2pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4c6'), network = network)}, Stockfish {utils.stockfish_eval('rnbqkbnr/pp2pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")
    print(f" 2. Qa4 Nc6 3.Qxc6: {eval_from_fen('r1bqkbnr/ppp1pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4c6'), network = network)}, Stockfish {utils.stockfish_eval('r1bqkbnr/ppp1pppp/2Qp4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")
    print(f" 2. Qa4 Nc6 3.Qxc6 bxc6: {eval_from_fen('r1bqkbnr/p1p1pppp/2pp4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3',last_move = chess.Move.from_uci('a4c6'), network = network)}, Stockfish {utils.stockfish_eval('r1bqkbnr/p1p1pppp/2pp4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3')}")
    print(f" 2. Qa4 Nd7 3.Qxd7: {eval_from_fen('r1bqkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4d7'), network = network)}, Stockfish {utils.stockfish_eval('r1bqkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")
    print(f" 2. Qa4 Nd7 3.Qxd7 Qxd7: {eval_from_fen('r1b1kbnr/pppqpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3',last_move = chess.Move.from_uci('a4d7'), network = network)}, Stockfish {utils.stockfish_eval('r1b1kbnr/pppqpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3')}")
    print(f" 2. Qa4 Bd7 3.Qxd7: {eval_from_fen('rn1qkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4d7'), network = network)}, Stockfish {utils.stockfish_eval('rn1qkbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")
    print(f" 2. Qa4 Bd7 3.Qxd7 Qxd7: {eval_from_fen('rn2kbnr/pppqpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3',last_move = chess.Move.from_uci('a4d7'), network = network)}, Stockfish {utils.stockfish_eval('rn2kbnr/pppqpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR w KQkq - 0 3')}")
    print(f" 2. Qa4 Qd7 3.Qxd7: {eval_from_fen('rnb1kbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2',last_move = chess.Move.from_uci('a4d7'), network = network)}, Stockfish {utils.stockfish_eval('rnb1kbnr/pppQpppp/3p4/8/2P5/8/PP1PPPPP/RNB1KBNR b KQkq - 0 2')}")


def bishop_test_cases_verbose(network = None):
    f_0 = 'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2'
    print(f" 1. d4 d6 2. e6 : {eval_from_fen(f_0, network = network)}, Stockfish {utils.stockfish_eval(f_0)}")
    f_1 = 'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3'
    print(f" 1. d4 d6 2. e6 Bb4: {eval_from_fen(f_1, network = network)}, Stockfish {utils.stockfish_eval(f_1)}")
    f_2 = 'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3'
    print(f" 1. d4 d6 2. e6 Bb4 3. c3: {eval_from_fen(f_2, network = network)}, Stockfish {utils.stockfish_eval(f_2)}")
    f_3 = 'rnbqk1nr/pppp1ppp/4p3/8/3PP3/2b5/PP3PPP/RNBQKBNR w KQkq - 0 4'
    print(f" 1. d4 d6 2. e6 Bb4 3. c3 Bxc3: {eval_from_fen(f_3, network = network)}, Stockfish {utils.stockfish_eval(f_3)}")
    return

def bishop_test_cases(network = None):
    fens = [
        'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2',
        'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3',
        'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3',
        'rnbqk1nr/pppp1ppp/4p3/8/3PP3/2b5/PP3PPP/RNBQKBNR w KQkq - 0 4',
        'rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2',
        'rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3',
        'rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3',
        'rnbqk1nr/pppp1ppp/8/4p3/3PP3/2b5/PP3PPP/RNBQKBNR w KQkq - 0 4']
    for f in fens:
        print(eval_from_fen(f, network = network))

def validation_examples(plot = False):
    val_data = Supervised_Dataset('data/win/debug/')
    data = DataLoader(val_data,batch_size = 1,shuffle = False, pin_memory = True, collate_fn= collate_fn, num_workers = 128)
    network = utils.load_network()
    network_evals = []
    stockfish_evals = []
    i = 0
    for inp,target in data:
        out = network(inp).detach()
        fen = fen_from_input(inp)
        stockfish_eval = train_utils.evaluate_board(chess.Board(fen),0.01)
        if stockfish_eval == None:
            stockfish_eval = train_utils.evaluate_board(chess.Board(fen),0.01)
        print(f'Network evaluation {out.item()}, Stockfish eval {stockfish_eval}')
        network_evals.append(out.item())
        stockfish_evals.append(stockfish_eval)
        print(f'Fen {fen}')
        i += 1
    print(f'Pearson correlation between Network and Stockfish {pearsonr(network_evals,stockfish_evals)}')

    stockfish_evals,network_evals = zip(*[(x,y) for x,y in sorted(zip(stockfish_evals,network_evals))])

    if plot:
        plt.plot(list(range(len(stockfish_evals))), stockfish_evals, color = 'blue', label = 'Stockfish')
        plt.plot(list(range(len(network_evals))), network_evals, color = 'red', label = 'Network')
        plt.ylim(0,1)
        plt.xlim(0,len(network_evals))
        plt.legend()
        plt.savefig('Evaluation_Comparison.png')


def look_at_data():
    dir = 'data/selfplay/1/train/'
    for file in os.listdir(dir):
        with open(dir+file, 'rb') as f:
            data = torch.load(f)
        print(data[0])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--fen',dest = 'fen')
    parser.add_argument('-m', '--move',dest = 'move')
    parser.add_argument('-n', '--network',dest = 'network')
    parser = parser.parse_args()
    path = 'networks/'+parser.network
    if parser.fen:
        if parser.move:
            move = chess.Move.from_uci(parser.move)
        else:
            move = None
        print(eval_from_fen(parser.fen, last_move= move, network = path))
    else:
        #juffi_test_cases(network = parser.network)
        #juffi_test_cases_notake(network = parser.network)
        #fixed_test_cases(network = parser.network)
        bishop_test_cases(network = path)


    
