from pstats import SortKey
from baselines.RandomBot import RandomBot
from Multiple_Board import MultipleBoard
from NN_Multiple_Board import NN_MultipleBoard
from Multiple_Board_Basic import MultipleBoardBasic
from baselines.trout import TroutBot
from reconchess import load_player, play_local_game, LocalGame
import cProfile
import time
import torch
from pstats import SortKey
from torch.multiprocessing import Process, set_start_method,Manager
import os
import numpy as np
from multiprocessing import Pool, set_start_method
from argparse import ArgumentParser, BooleanOptionalAction

def one_self_play(player,output_path, export_game = False):
    player_1, player_2 = player
    print('are we here?')
    player_1 = NN_MultipleBoard(player_1, output_path = output_path, randomized= True)
    player_2 = NN_MultipleBoard(player_2, output_path = output_path, randomized= True)
    game = LocalGame(seconds_per_player= 900)
    res = play_local_game(player_1,player_2,game)
    result = res[0]
    del res, game, player_1, player_2
    print('Finished this one game')
    return result

def test():
    print(1)

def one_play(players, export_game = False):
    player_1, player_2 = players
    player_1 = NN_MultipleBoard(player_1)
    player_1.save_csv = True
    player_2 = player_2()
    game = LocalGame(seconds_per_player= 900)
    if np.random.rand() > 0.5:
        res = play_local_game(player_1,player_2,game)
        p1_won = res[0]
    else:
        res = play_local_game(player_2,player_1,game)
        p1_won = not res[0]

    if export_game:
        res[2].save('game.json')
    return p1_won

def create_samples(num, path,out_path):
    parallel = 10
    with Pool(processes=parallel) as p:
        while len(os.listdir(out_path)) < num:
                p.apply_async(one_self_play, args = [(path,path),out_path,False])

result_list = []
def log_result(results):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.extend(results)
    print(result_list)


def test_winrate(path_1, path_2, num_games = 10):
    pool_size = 10
    while len(result_list) < 10:
        print('New iter')
        p = Pool(pool_size)
        result_list.extend(p.starmap(one_self_play,[[(path_1,path_2),None] for _ in range(10)]))
    p1_wins = np.mean(result_list)
    print(f'Player 1 wins: {p1_wins}')
    print(f'Player 2 wins: {1-p1_wins}')

if __name__ == '__main__':
    default_path = 'networks/stockfish/tuned.pt'
    parser = ArgumentParser()
    parser.add_argument('-s','--samples', dest = 'samples', type = int)
    parser.add_argument('-o','--outpath', dest = 'outpath')
    parser.add_argument('-n','--network', dest = 'network')
    parser.add_argument('-w','--winrate', dest = 'winrate')

    parser = parser.parse_args()
    if parser.samples is None and parser.winrate is False:
        #one_play(('networks/rbc_2nd_200/rbc_2nd_200_v1.pt',TroutBot), export_game = True)
        output_path = None
        create_samples(10000,default_path)
    elif parser.samples is not None:
        output_path = parser.outpath + '/'
        create_samples(parser.samples, parser.network, output_path)
        print(f'Using {parser.network}')
    elif parser.winrate:
        p1, p2 = parser.winrate.split(' ')
        test_winrate(p1,p2, num_games = 10)
