from pstats import SortKey
from baselines.RandomBot import RandomBot
from Multiple_Board import MultipleBoard
from baselines.trout import TroutBot
from baselines.RandomBot import RandomBot
from baselines.agressive_tree import AggressiveTree
from baselines.agressive_horse import AggressiveHorse
from Multiple_Board_Basic import MultipleBoardBasic
from baselines.stocky_bot import StockyInference
from reconchess import load_player, play_local_game, LocalGame
import cProfile
from multiprocessing import Pool
import numpy as np
import csv

def play_game(opponent):
    player_1 = MultipleBoard()
    player_2 = opponent()

    game = LocalGame(seconds_per_player= 900)
    res = play_local_game(player_1,player_2,game)
    if res[0] == True:
        result_white = 1
    else:
        result_white = 0

    

    player_1 = opponent()
    player_2 = MultipleBoard()

    game = LocalGame(seconds_per_player= 900)
    res = play_local_game(player_1,player_2,game)
    if res[0] == True:
        result_black = 0
    else:
        result_black = 1

    return result_white,result_black


pool_size = 25
number_of_games = 50
results_white = []
results_black = []
opponents = [RandomBot,AggressiveTree,AggressiveHorse,MultipleBoardBasic,StockyInference]

for opponent in opponents:
    results_white = []
    results_black = []
    print(f'Going to play {number_of_games} games against {str(opponent())}')
    for _ in range(int(number_of_games/pool_size)):
        with Pool(pool_size) as p:
            results = p.map(play_game,[opponent]*pool_size)
            for w,b in results:
                results_white.append(w)
                results_black.append(b)
        

    print(f'Winrate as white: {np.mean(results_white)} \nWinrate as black: {np.mean(results_black)}')
    with open(f'testing_results/{str(opponent())}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['White'])
        writer.writerow(results_white)
        writer.writerow(['Black'])
        writer.writerow(results_black)
