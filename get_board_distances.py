from argparse import ArgumentParser
import requests
import json
import time
from create_dataset import get_all_observations
from create_data import get_all_white_boards, get_all_black_boards
import matplotlib.pyplot as plt
import sys


 
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                help="JSON file to parse")
parser.add_argument("-o", "--online", dest="online",
                help="Online file to parse")
parser.add_argument("-c", "--color", dest="color",
                help="Color to get stats of")
parser.add_argument("-t", "--times", dest="times",
                help="Individual times for parts")
args = parser.parse_args()



if args.filename:
    path = ''
    start_time = time.time()
    with open(args.filename, 'r') as f:
        data = json.load(f)
if args.online:
    path = ''
    start_time = time.time()
    url = f'https://rbc.jhuapl.edu/api/games/{args.online}/game_history/download'
    data = requests.get(url).json()
if args.color is None and args.filename is None and args.online is None:
    print('Arguments not set, defaulting')
    args.color = 'black'
    args.online = 598023
    url = f'https://rbc.jhuapl.edu/api/games/{args.online}/game_history/download'
    data = requests.get(url).json()




    
if args.times is None:
    args.times = True
else:
    args.times = bool(args.times)

if args.color is None:
    white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations = get_all_observations(data)
elif args.color == 'white':
    white_sense_boards,white_move_boards,white_true_boards,agent = get_all_white_boards(data, as_path = False, all_things = True)
elif args.color == 'black':
    black_sense_boards,black_move_boards,black_true_boards,agent = get_all_black_boards(data, as_path = False, all_things = True)
else:
    raise Exception
if args.times:
    labels = agent.times.keys()
    last_sense = 0
    last_handle = 0
    last_move = 0
    for i,(handle,sense,move) in enumerate(zip(*agent.times.values())):
        plt.bar(labels,[handle,sense,move],bottom = [last_handle,last_sense,last_move], label = i)
        last_sense += sense
        last_handle += handle
        last_move += move
    #plt.legend()    
    print(len(agent.times['Handle opponent move']),len(agent.times['Choose sense']),len(agent.times['Choose move']))
    for k,v in agent.times.items():
        print(f'{k} : Sum {sum(v)}')
    plt.savefig('debugging/times.png')
