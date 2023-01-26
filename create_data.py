from fileinput import filename
import requests
from Multiple_Board import MultipleBoard
from NN_Multiple_Board import NN_MultipleBoard
from ray.util.multiprocessing import Pool
from models import Siamese_RBC_dataset
import tqdm
from argparse import ArgumentParser
from create_dataset import get_all_observations
import models
import sys
from collections import defaultdict
import csv
import pandas as pd
import os
import tqdm
import torch
import chess
import multiprocessing as mp
import numpy as np
import pickle
import time
import lzma
import json

def get_all_white_boards(path, as_path = True, all_things = False):
    sense_boards = []
    move_boards = []
    true_boards = []
    start_time = time.time()
    all_seconds = 900
    if as_path:
        with open(path) as f:
            data = json.load(f)
            f.close()
    else:
        data = path
    agent = NN_MultipleBoard(normal = False)
    agent.save_csv = True
    agent.handle_game_start(True, data['fens_before_move']['true'][0],data['black_name'])
    agent.handle_opponent_move_result(None, None)
    for turn in range(len(data['senses']['true'])):
        sense_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        if all_things:
            agent.choose_sense(chess.SQUARES, list(chess.Board(data['fens_before_move']['true'][turn]).pseudo_legal_moves), all_seconds-(time.time()-start_time))
        else:
            agent.siamese.last_sense = data['senses']['true'][turn]
        agent.handle_sense_result(data['sense_results']['true'][turn])
        if all_things:
            agent.choose_move(list(chess.Board(data['fens_before_move']['true'][turn]).pseudo_legal_moves), all_seconds-(time.time()-start_time))

        if turn >= len(data['fens_before_move']['true']):
            break

        true_boards.append(data['fens_before_move']['true'][turn])
        if agent.board_dict.size() == 0:
            break
        move_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        requested_move = chess.Move.from_uci(data['requested_moves']['true'][turn]['value']) if data['requested_moves']['true'][turn] else None
        taken_move = chess.Move.from_uci(data['taken_moves']['true'][turn]['value']) if data['taken_moves']['true'][turn] else None
        agent.handle_move_result(requested_move,taken_move,\
            True if data['capture_squares']['true'][turn] != None else False, data['capture_squares']['true'][turn])
        if agent.board_dict.size() == 0:
            break
        if len(data['capture_squares']['false']) > turn:
            agent.handle_opponent_move_result(True if data['capture_squares']['false'][turn] != None else False, data['capture_squares']['false'][turn])
        
            if agent.board_dict.size() == 0:
                break
    
    #agent.handle_opponent_move_result(False, None)
    del data
    return sense_boards,move_boards,true_boards,agent

def get_all_black_boards(path,as_path = True,all_things = False):
    sense_boards = []
    move_boards = []
    true_boards = []
    all_seconds = 900
    start_time = time.time()
    if as_path:
        with open(path) as f:
            data = json.load(f)
            f.close()
    else:
        data = path

    agent = NN_MultipleBoard(normal = False)
    agent.save_csv = True
    agent.handle_game_start(False, data['fens_before_move']['false'][0],data['white_name'])
    
    for turn in range(len(data['taken_moves']['false'])):
        true_boards.append(data['fens_before_move']['false'][turn])
        agent.handle_opponent_move_result(True if data['capture_squares']['true'][turn] != None else False, data['capture_squares']['true'][turn])
        
        if agent.board_dict.size() == 0:
            break
        sense_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        if all_things:
            agent.choose_sense(chess.SQUARES, list(chess.Board(data['fens_before_move']['false'][turn]).pseudo_legal_moves), all_seconds-(time.time()-start_time))
        agent.siamese.last_sense = data['senses']['false'][turn]
        agent.handle_sense_result(data['sense_results']['false'][turn])

        if agent.board_dict.size() == 0:
            break
        move_boards.append([board.fen() for board in agent.board_dict.get_boards()])
        if all_things:
            agent.choose_move(list(chess.Board(data['fens_before_move']['false'][turn]).pseudo_legal_moves), all_seconds-(time.time()-start_time))
        requested_move = chess.Move.from_uci(data['requested_moves']['false'][turn]['value']) if data['requested_moves']['false'][turn] else None
        taken_move = chess.Move.from_uci(data['taken_moves']['false'][turn]['value']) if data['taken_moves']['false'][turn] else None
        agent.handle_move_result(requested_move,taken_move,\
            True if data['capture_squares']['false'][turn] != None else False, data['capture_squares']['false'][turn])

        if agent.board_dict.size() == 0:
            break
    
    if len(data['taken_moves']['false']) > len(data['senses']['false']):
        agent.siamese.last_sense = data['senses']['false'][-1]
        agent.handle_sense_result(data['sense_results']['false'][-1])
        agent.choose_move(list(chess.Board(data['fens_before_move']['false'][-1]).pseudo_legal_moves), 1000)

    del data
    return sense_boards,move_boards,true_boards,agent

def process_files(file = None, online = None, outpath = 'data/siamese_playerlabel'):
    if file is not None:
        with open(path+file) as f:
            data = json.load(f)
            if len(data['fens_before_move']['true']) == 0 or len(data['fens_before_move']['false']) == 0:
                return
    if online is not None:
        url = f'https://rbc.jhuapl.edu/api/games/{online}/game_history/download'
        data = requests.get(url).json()

    finished_path = outpath+'/finished_files/'
    if file is not None:
        if not os.path.exists(finished_path):
            os.mkdir(finished_path)
        if os.path.exists(finished_path+file):
            print(f'Skipping file {file}')
            return
    if not os.path.exists(outpath+'/move/'):
        os.mkdir(outpath+'/move/')
    if not os.path.exists(outpath+'/sense/'):
        os.mkdir(outpath+'/sense/')
    white_name = data['white_name']
    black_name = data['black_name']
    white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations = get_all_observations(data)
    white_sense_boards,white_move_boards,white_true_boards,_ = get_all_white_boards(data, as_path = False)
    black_sense_boards,black_move_boards,black_true_boards,_ = get_all_black_boards(data, as_path= False)
    white_sense_num = min(len(white_sense_boards),len(white_true_boards),len(white_presense_observations))
    white_move_num = min(len(white_move_boards),len(white_true_boards),len(white_premove_observations))
    black_sense_num = min(len(black_sense_boards),len(black_true_boards),len(black_presense_observations))
    black_move_num = min(len(black_move_boards),len(black_true_boards),len(black_premove_observations))



    for boards,true_board,obs in zip(white_sense_boards[1:white_sense_num], white_true_boards[1:white_sense_num],white_presense_observations[1:white_sense_num]):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/sense/{time.time()}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, black_name), f)
                f.close()
    for boards,true_board,obs in zip(white_move_boards[1:white_move_num], white_true_boards[1:white_move_num],white_premove_observations[1:white_move_num]):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/move/{time.time()}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, black_name), f)
                f.close()

    for boards,true_board,obs in zip(black_sense_boards[:black_sense_num], black_true_boards[:black_sense_num],black_presense_observations[:black_sense_num]):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/sense/{time.time()}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards, white_name), f)
                f.close()
    for boards,true_board,obs in zip(black_move_boards[:black_move_num], black_true_boards[:black_move_num],black_premove_observations[:black_move_num]):
        if len(boards) > 1:
            with lzma.open(f'{outpath}/move/{time.time()}.pt', 'wb') as f:
                pickle.dump((obs,true_board,boards,white_name), f)
                f.close()
    if file is not None:
        open(finished_path+file, 'w')
    del file, data, white_sense_boards,white_true_boards,white_presense_observations,white_premove_observations,black_sense_boards,black_true_boards,black_presense_observations,black_premove_observations

def debug_files():
    path = '/home/fawler/tbertram/RBC/siamese/games/game/'
    file = os.listdir(path)[0]
    white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations = get_all_observations(path+file)
    white_sense_boards,white_move_boards,white_true_boards = get_all_white_boards(path+file)
    #black_sense_boards,black_move_boards,black_true_boards = get_all_black_boards(path+file)
    white_sense_num = min(len(white_sense_boards),len(white_true_boards),len(white_presense_observations))
    white_move_num = min(len(white_move_boards),len(white_true_boards),len(white_premove_observations))
    #black_sense_num = min(len(black_sense_boards),len(black_true_boards),len(black_presense_observations))
    #black_move_num = min(len(black_move_boards),len(black_true_boards),len(black_premove_observations))

    print(len(white_sense_boards))
    print(len(white_true_boards))
    print(len(white_presense_observations))
    for i,(boards,true_board,obs) in enumerate(zip(white_sense_boards, white_true_boards,white_presense_observations)):
        
        padded_obs = torch.zeros(20,90,8,8)
        len_unpadded_anchor = obs.size(0)
        padded_obs[-len_unpadded_anchor:,:,:,:] = obs
        padded_obs = padded_obs.view(1,20*90,8,8)

        with open(f'{path}sense_obs_{i}.csv','w') as f:
            writer = csv.writer(f,delimiter = ';')
            writer.writerows(padded_obs)
        with open(f'{path}sense_boards_{i}.csv','w') as f:
            writer = csv.writer(f,delimiter = ';')
            for b in boards:
                writer.writerows(b)
    for i,(boards,true_board,obs) in enumerate(zip(white_move_boards, white_true_boards,white_premove_observations)):
        
        padded_obs = torch.zeros(20,90,8,8)
        len_unpadded_anchor = obs.size(0)
        padded_obs[-len_unpadded_anchor:,:,:,:] = obs
        padded_obs = padded_obs.view(1,20*90,8,8)
        with open(f'{path}move_obs_{i}.csv','w') as f:
            writer = csv.writer(f,delimiter = ';')
            writer.writerows(padded_obs)
        with open(f'{path}move_boards_{i}.csv','w') as f:
            writer = csv.writer(f,delimiter = ';')
            for b in boards:
                writer.writerows(b)



def split_data(path):
    train_path = path+'train/'
    test_path = path+'val/'
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    files = os.listdir(path)
    len_test = len(files)//10
    test_files = np.random.choice(files,len_test)
    for file in files:
        if not '.pt' in file:
            continue
        if file in test_files:
            os.rename(path+file,test_path+file)
        else:
            os.rename(path+file,train_path+file)

def remove_single_boards(full_path):
    with lzma.open(full_path, 'rb') as f:
        data = pickle.load(f)

    if len(data[2]) == 1:
        os.remove(full_path)
    else:
        new_fens = []
        positive_board = Siamese_RBC_dataset.board_from_fen(None,data[1])
        for fen in data[2]:
            board = Siamese_RBC_dataset.board_from_fen(None,fen)
            if not torch.equal(positive_board,board):
                new_fens.append(fen)
    
        if len(new_fens) > 0:        
            with lzma.open(full_path, 'wb') as f:
                pickle.dump((data[0],data[1],new_fens),f)
        else:
            os.remove(full_path)


piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

def board_from_fen(board):
    tensor_board = torch.zeros(12,8,8)
    board = chess.Board(board)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
    return tensor_board

def remove_duplicates(path):
    files = os.listdir(path)
    out_path = f'{path}no_dupes/'
    anchors = []
    unpadded_anchors = []
    positives = []
    negatives = []
    for file in files:
        if '.pt' not in file:
            continue
        with lzma.open(path+file, 'rb') as f:
            data = pickle.load(f)
        anchor,positive,negative = data
        padded_anchor = torch.zeros(20,90,8,8)
        len_unpadded_anchor = anchor.size(0)
        padded_anchor[-len_unpadded_anchor:,:,:,:] = anchor
        padded_anchor = padded_anchor.view(20*90,8,8)
        anchors.append(padded_anchor)
        unpadded_anchors.append(anchor)
        positives.append(board_from_fen(positive)),
        negatives.append(negative)
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    
    choices = torch.cat((anchors,positives),dim=1)
    unique_choices,choice_indizes = torch.unique(choices,dim=0, return_inverse = True)
    done_indizes = []
    for i in range(choice_indizes.size(0)):
        if choice_indizes[i] not in done_indizes:
            with lzma.open(out_path+files[i], 'wb') as f:
                pickle.dump((unpadded_anchors[i],positives[i],negatives[i]), f)
                f.close()
        done_indizes.append(choice_indizes[i])

def bot_numbers(path):
    numbers = defaultdict(int)

    for file in tqdm.tqdm(os.listdir(path)):
        with open(path+file) as f:
            data = json.load(f)
            numbers[data['white_name']] += 1
            numbers[data['black_name']] += 1
    sorted_opponents = sorted(numbers, key=numbers.get, reverse=True)
    with open('game_numbers.csv', 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        for k in sorted_opponents:
            writer.writerow([k, numbers[k]])

def convert_data_to_everything(path,filename,outpath,encoding,empty_encoding):
    try:
        with lzma.open(path+filename, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(path)
        print(e)
        pass
    else:
        anchor = data[0]
        positive = board_from_fen(data[1])
        if data[2]:
            num_choices = min(len(data[2]),100)
            negatives = [board_from_fen(b) for b in np.random.choice(data[2],num_choices)]
            negatives = [n for n in negatives if not torch.equal(n,positive)]
            
            if data[3] in encoding:
                finished_encoding = encoding[data[3]]
            else:
                finished_encoding = empty_encoding


    padded_anchor = torch.zeros(20,90,8,8)
    len_unpadded_anchor = anchor.size(0)
    padded_anchor[-len_unpadded_anchor:,:,:,:] = anchor
    padded_anchor = padded_anchor.view(20*90,8,8)
    negative_length = len(negatives)
    with open(outpath+filename,'wb') as f:
        torch.save((anchor,positive,negatives,negative_length,finished_encoding),f)



    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                    help="JSON file to parse", metavar="FILE")
    parser.add_argument("-s", "--split", dest="split",
                    help="Whether to split train test ", metavar="FILE")

    args = parser.parse_args()
    path = 'data/games/'

    # process_files(online = '619027', outpath= 'data/tmp_data/')
    
    if args.filename:
        url = f'https://rbc.jhuapl.edu/api/games/{args.filename}/game_history/download'
        response = requests.get(url).json()
        get_all_observations(args.filename)
    elif args.split:
        split_data(args.split)
    else:
        files = os.listdir(path)
        pool = Pool()
        results = pool.map(process_files, files)
        pool.close()
        pool.join()


        


        # files = os.listdir(path)
        # df = pd.DataFrame(files)
        # df.to_csv('files_to_do.csv')
        # active_processes = []
        # time_since_last_save = time.time()
        # files_left = df.shape[0]
        # print(f'Files left to do: {files_left}')
        # while df.shape[0] > 0:
        #     while len(active_processes) < 1:
        #         next_file = df.iloc[0]
        #         p = mp.Process(target = process_files,args = (next_file))
        #         df = df.iloc[1:]
        #         active_processes.append(p)
        #         p.start()
                  
        #         #df.to_csv('files_to_do.csv')
        #     for p in active_processes:
        #         p.join(timeout = 0)
        #         if not p.is_alive():
        #             p.join()
        #             active_processes.remove(p)
        #         else:
        #             time.sleep(1)f
        #             print(p, p.is_alive())
        #     if time.time() - time_since_last_save >= 60:
        #         time_since_last_save = time.time()
        #         df.to_csv('files_to_do.csv')
        #         new_file_amount = df.shape[0]
        #         print(f'Files left to do: {new_file_amount}')
        #         print(f'Processed: {files_left - new_file_amount} files in the last 1 minutes')
        #         files_left = new_file_amount

    