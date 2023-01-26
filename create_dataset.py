from collections import defaultdict
from itertools import count
import json
import os
import tqdm
import sys
import requests
import time
import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import int_to_row_column, row_column_to_int
import csv
import chess


def sense_location_to_area(location):
    area = torch.zeros(8,8)
    row,column = int_to_row_column(location)

    for r in [-1,0,1]:
        for c in [-1,0,1]:
            if row+r < 0 or row+r > 7 or column+c < 0 or column+c > 7:
                continue
            area[row+r,column+c] += 1
    return area

def fill_codes(codes,borders):
    i = 0
    for nSquares in range(1,8):
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            if direction == 'N':
                borders[i,nSquares:,:] = 1
            if direction == 'NE':
                borders[i,nSquares:,:-nSquares] = 1
            if direction == 'E':
                borders[i,:,:-nSquares] = 1
            if direction == 'SE':
                borders[i,:-nSquares,:-nSquares] = 1
            if direction == 'S':
                borders[i,:-nSquares,:] = 1
            if direction == 'SW':
                borders[i,:-nSquares,nSquares:] = 1
            if direction == 'W':
                borders[i,:,nSquares:] = 1
            if direction == 'NW':
                borders[i,nSquares:,nSquares:] = 1
            codes[(nSquares,direction)] = i
            i += 1
    for two in ["N","S"]:
        for one in ["E","W"]:
            if two == 'S' and one == 'E':
                borders[i,:-2,:-1] = 1
            if two == 'N' and one == 'E':
                borders[i,2:,:-1] = 1
            if two == 'S' and one == 'W':
                borders[i,:-2,1:] = 1
            if two == 'N' and one == 'W':
                borders[i,2:,1:] = 1
            codes[("knight", two, one)] , i = i , i + 1
    for two in ["E","W"]:
        for one in ["N","S"]:
            if two == 'E' and one == 'N':
                borders[i,1:,:-2] = 1
            if two == 'W' and one == 'S':
                borders[i,:-1,2:] = 1
            if two == 'E' and one == 'S':
                borders[i,:-1,:-2] = 1
            if two == 'W' and one == 'N':
                borders[i,1:,2:] = 1
            codes[("knight", two, one)] , i = i , i + 1
    for move in ["N","NW","NE"]:
        for promote_to in ["Rook","Knight","Bishop"]:
            borders[i,1,:] = 1
            borders[i,6,:] = 1
            codes[("underpromotion", move, promote_to)] , i = i , i + 1
    return codes,borders

def move_to_location(move,board):
    starting_square = chess.SQUARES[chess.parse_square(move[:2])]
    piece = str(board.piece_at(starting_square))
    end_square = chess.SQUARES[chess.parse_square(move[2:4])]
    start_row = 7-chess.square_rank(starting_square)
    end_row = 7-chess.square_rank(end_square)
    start_col = chess.square_file(starting_square)
    end_col = chess.square_file(end_square)
    promotion = move[4] if len(move) == 5 else None

    row_dif = start_row-end_row
    col_dif = start_col-end_col
    directions = {'S': -row_dif if row_dif < 0 else 0,'N': row_dif if row_dif > 0 else 0,
    'E': -col_dif if col_dif < 0 else 0,'W': col_dif if col_dif > 0 else 0}
    if piece == 'N' or piece == 'n':
        for k,v in directions.items():
            if v == 2:
                first_coordinate = k
            elif v == 1:
                second_coordinate = k
        return codes['knight',first_coordinate,second_coordinate],start_row,start_col
    else:
        if promotion and promotion != 'q':
            done_directions = [d for d in directions.keys() if directions[d] > 0]
            if 'E' in done_directions:
                dir = 'NE'
            elif 'W' in done_directions:
                dir = 'NW'
            else:
                dir = 'N'

            if promotion == 'n':
                promote_to = 'Knight'
            elif promotion == 'r':
                promote_to = 'Rook'
            elif promotion == 'b':
                promote_to = 'Bishop'
            return codes["underpromotion",dir,promote_to],start_row,start_col
        else:
            direction = ''
            move_length = None
            if directions['N'] > 0:
                direction += 'N'
                move_length = directions['N']
            if directions['S'] > 0:
                direction += 'S'
                assert not move_length or move_length == directions['S']
                move_length = directions['S']
            if directions['E'] > 0:
                direction += 'E'
                assert not move_length or move_length == directions['E']
                move_length = directions['E']
            if directions['W'] > 0:
                direction += 'W'
                assert not move_length or move_length == directions['W']
                move_length = directions['W']
            return codes[(move_length,direction)],start_row,start_col

def location_to_move(location, print_out = False):
    try:
        move_type = inverse_codes[location[0]]
    except Exception as e:
        print(inverse_codes)
        print(location[0])
        print(inverse_codes[location[0]])
        print(type(e))
        print(location)
        print(f'Exception with move type')
        raise Exception
    start_col = location[2]
    end_col = start_col
    start_row = location[1]
    end_row = start_row
    start_square = col_to_int[start_col]+row_to_int[start_row]
    piece_promotion = ''
    if move_type[0] == 'knight':
        if move_type[1] == 'S':
            if move_type[2] == 'E':
                end_col = start_col + 1
                end_row = start_row + 2
            else:
                end_col = start_col - 1
                end_row = start_row + 2
        if move_type[1] == 'N':
            if move_type[2] == 'E':
                end_col = start_col + 1
                end_row = start_row - 2
            else:
                end_col = start_col - 1
                end_row = start_row - 2
        if move_type[1] == 'E':
            if move_type[2] == 'N':
                end_col = start_col + 2
                end_row = start_row - 1
            else:
                end_col = start_col + 2
                end_row = start_row + 1
        if move_type[1] == 'W':
            if move_type[2] == 'N':
                end_col = start_col - 2
                end_row = start_row - 1
            else:
                end_col = start_col - 2
                end_row = start_row + 1
    else:
        move_length = move_type[0]
        if move_length == 'underpromotion':
            if move_type[2] == 'Rook':
                piece_promotion = 'r'
            elif move_type[2] == 'Bishop':
                piece_promotion = 'b'
            elif move_type[2] == 'Knight':
                piece_promotion = 'n'
            else:
                print(move_type[2])
                raise Exception
            if start_row == 6:
                end_row = 7
            else:
                end_row = 0
            if 'W' in move_type[1]:
                if start_row == 6:
                    end_col = start_col -1
                else:
                    end_col = start_col - 1
            if 'E' in move_type[1]:
                if start_row == 6:
                    end_col = start_col +1
                else:
                    end_col = start_col + 1
        else:
            if 'N' in move_type[1]:
                end_row = start_row - move_length
            if 'S' in move_type[1]:
                end_row = start_row + move_length
            if 'E' in move_type[1]:
                end_col = start_col + move_length
            if 'W' in move_type[1]:
                end_col = start_col - move_length
    try:
        end_square = col_to_int[end_col]+row_to_int[end_row]
    except Exception as e:
        print(location)
        print(e)
        print(type(e), end_col, end_row, move_type)
        print(f'Move out of board; start {start_square}, {move_type}, start/end col: {start_col}/{end_col}, start/end row: {start_row}/{end_row}')
        return None
    move = start_square+end_square
    if print_out:
        print(f'No error with move {move}, {move_type}, start/end col: {col_to_int[start_col]}/{col_to_int[end_col]}, start/end row: {row_to_int[start_row]}/{row_to_int[end_row]}')
    return move+piece_promotion

def count_opponents():
    path = '/home/fawler/tbertram/RBC/scraper/data/'
    opponents = defaultdict(int)
    for file in tqdm.tqdm(os.listdir(path)):
        if 'json' not in file:
            continue
        else:
            with open(path+file) as f:
                data = json.load(f)
            opponents[data['white_name']] += 1
            opponents[data['black_name']] += 1
    sorted_opponents = sorted(opponents, key=opponents.get, reverse=True)
    with open('opponent_counts.csv','w') as f:
        writer = csv.writer(f,delimiter = ';')
        for opp in sorted_opponents:
            writer.writerow([opp,opponents[opp]])
            
    plt.plot(sorted_opponents[:100],[opponents[opp] for opp in sorted_opponents[:100]])
    plt.savefig('opp_count.pdf')
    plt.show()
    over_1000_games = []
    for opp in sorted_opponents:
        if opponents[opp] >= 1000:
            over_1000_games.append(opp)
        else:
            break
    over_1000_games.remove('random')
    return over_1000_games

def parse_to_pgn():
    path = '/home/fawler/tbertram/RBC/scraper/data/'
    out_file = 'rbc_results.pgn'
    with open(out_file, 'w') as f:
        for file in tqdm.tqdm(os.listdir(path)):
            if 'json' not in file:
                continue
            else:
                with open(path+file) as j:
                    data = json.load(j)
                
                p1 = data['white_name']
                p2 = data['black_name']
                p1_out = f'[White "{p1}"]'
                p2_out = f'[Black "{p2}"]'
                winner = '1-0' if data['winner_color'] else '0-1'
                winner_out = f'[Result "{winner}"]'
                tmp_game = ' 1. d4 d5'
                f.write(p1_out+p2_out+winner_out+tmp_game+'\n')

def process_file(file,possible_opponents):
    len_one_stack = 90
    path = '/home/fawler/tbertram/RBC/data/games/'
    outpath_sense = '/home/fawler/tbertram/RBC/data/sense/'
    outpath_move = '/home/fawler/tbertram/RBC/data/move/'
    white_input_history = []
    black_input_history = []

    with open(path+file) as f:
        data = json.load(f)
    white_sense_results = data['sense_results']['true']
    white_player = data['white_name']
    white_senses = data['senses']['true']
    white_capture_squares = data['capture_squares']['true']
    white_requested_moves = data['requested_moves']['true']
    white_taken_moves = data['taken_moves']['true']
    white_fens_before = data['fens_before_move']['true']

    
    black_sense_results = data['sense_results']['false']
    black_player = data['black_name']
    black_senses = data['senses']['false']
    black_capture_squares = data['capture_squares']['false']
    black_requested_moves = data['requested_moves']['false']
    black_taken_moves = data['taken_moves']['false']
    black_fens_before = data['fens_before_move']['false']
    for i,(sense,result,_) in enumerate(zip(white_senses,white_sense_results,white_capture_squares)):     
        target_result = torch.ones(1)
        if not data['winner_color']:
            target_result *= -1      
            won_or_not = 'loss'  
        else:  
            won_or_not = 'won'
        #0: opponents capture
        #1-73: last requested/taken move
        #74: last move captured a piece
        #75: last move was None
        #76-81: own pieces
        #82: sensed area
        #83-88: sensed result
        #last of stack: color
        board = torch.zeros(len_one_stack,8,8)
        board[-1,:,:] = 1

        #fill in last opponent capture if exists
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = white_requested_moves[i-1]['value'] if white_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if white_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
            
        #fill in own board
        own_pieces = chess.Board(white_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        white_input_history.append(board)
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1
            torch.save((torch.stack(white_input_history[-20:]),target_sense.view(64),target_pieces,target_result),f'{outpath_sense}file_{time.time()}_{white_player}_white_{won_or_not}.pt')
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if not c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = white_requested_moves[i]['value'] if white_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        white_input_history[-1] = board
        torch.save((torch.stack(white_input_history[-20:]),target_move,target_pieces,target_result),f'{outpath_move}file_{time.time()}_{white_player}_white_{won_or_not}.pt')
    for i,(sense,result,_) in enumerate(zip(black_senses,black_sense_results,black_capture_squares)):
        target_result = torch.ones(1)
        if data['winner_color']:
            target_result *= -1
            won_or_not = 'loss'
        else:
            won_or_not = 'win'
        board = torch.zeros(len_one_stack,8,8)

        #fill in last opponent capture if exists
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = black_requested_moves[i-1]['value'] if black_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if black_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
        
        #fill in own board
        own_pieces = chess.Board(black_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if not piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        black_input_history.append(board)    

        
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1  
            torch.save((torch.stack(black_input_history[-20:]),target_sense.view(64),target_pieces,target_result),f'{outpath_sense}file_{time.time()}_{black_player}_black_{won_or_not}.pt')
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = black_requested_moves[i]['value'] if black_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        black_input_history[-1] = board
        torch.save((torch.stack(black_input_history[-20:]),target_move,target_pieces,target_result),f'{outpath_move}file_{time.time()}_{black_player}_black_{won_or_not}.pt')


def process_file_win(file = None,history = None, color = None, outpath_win = None):
    path = 'data/games/'
    if not outpath_win:
        outpath_win = '/home/fawler/tbertram/RBC/data/selfplay/'


    if history is not None:
        tmp_path = f'{outpath_win}tmp.json'
        history.save(tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        os.remove(tmp_path)
        

    if file is not None:
        with open(path+file) as f:
            data = json.load(f)
        
    white_player = data['white_name']
    white_boards = data['fens_before_move']['true']
    white_capture_squares = [None]+data['capture_squares']['true'][:-1]
    win = torch.ones(1) if data['winner_color'] else torch.zeros(1)

    
    black_player = data['black_name']
    black_boards = data['fens_before_move']['false']
    black_capture_squares = [None]+data['capture_squares']['false'][:-1]

    if color == True:
        for white_board,white_capture_square in zip(white_boards,white_capture_squares):
            torch.save((white_board,white_capture_square,torch.ones((1,8,8)),win),f'{outpath_win}file_{time.time()}_{white_player}_white_{bool(win)}.pt')
    elif color == False:
        for black_board,black_capture_square in zip(black_boards,black_capture_squares):
            torch.save((black_board,black_capture_square,torch.zeros((1,8,8)),win),f'{outpath_win}file_{time.time()}_{black_player}_black_{bool(win)}.pt')
    else:
        for white_board,white_capture_square in zip(white_boards,white_capture_squares):
            torch.save((white_board,white_capture_square,torch.ones((1,8,8)),win),f'{outpath_win}file_{time.time()}_{white_player}_white_{bool(win)}.pt')
        for black_board,black_capture_square in zip(black_boards,black_capture_squares):
            torch.save((black_board,black_capture_square,torch.zeros((1,8,8)),win),f'{outpath_win}file_{time.time()}_{black_player}_black_{bool(win)}.pt')


def get_all_observations(data, as_path = False):
    len_one_stack = 90
    white_input_history = []
    black_input_history = []

    white_presense_observations = []
    white_premove_observations = []
    black_presense_observations = []
    black_premove_observations = []
    
    white_sense_results = data['sense_results']['true']
    white_player = data['white_name']
    white_senses = data['senses']['true']
    white_capture_squares = data['capture_squares']['true']
    white_requested_moves = data['requested_moves']['true']
    white_taken_moves = data['taken_moves']['true']
    white_fens_before = data['fens_before_move']['true']

    
    black_sense_results = data['sense_results']['false']
    black_player = data['black_name']
    black_senses = data['senses']['false']
    black_capture_squares = data['capture_squares']['false']
    black_requested_moves = data['requested_moves']['false']
    black_taken_moves = data['taken_moves']['false']
    black_fens_before = data['fens_before_move']['false']
    for i,(sense,result,_) in enumerate(zip(white_senses,white_sense_results,white_capture_squares)):     
        target_result = torch.ones(1)
        if not data['winner_color']:
            target_result *= -1      
            won_or_not = 'loss'  
        else:  
            won_or_not = 'won'
        #0: opponents capture
        #1-73: last requested/taken move
        #74: last move captured a piece
        #75: last move was None
        #76-81: own pieces
        #82: sensed area
        #83-88: sensed result
        #last of stack: color
        board = torch.zeros(len_one_stack,8,8)
        board[-1,:,:] = 1

        #fill in last opponent capture if exists
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = white_requested_moves[i-1]['value'] if white_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if white_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
            
        #fill in own board
        own_pieces = chess.Board(white_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        white_input_history.append(board)
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1
            white_presense_observations.append(torch.clone(torch.stack(white_input_history[-20:])))
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if not c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = white_requested_moves[i]['value'] if white_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        white_input_history[-1] = board
        white_premove_observations.append(torch.clone(torch.stack(white_input_history[-20:])))
    for i,(sense,result,_) in enumerate(zip(black_senses,black_sense_results,black_capture_squares)):
        target_result = torch.ones(1)
        if data['winner_color']:
            target_result *= -1
            won_or_not = 'loss'
        else:
            won_or_not = 'win'
        board = torch.zeros(len_one_stack,8,8)

        #fill in last opponent capture if exists
        if i > 0:
            if white_capture_squares[i-1] is not None:
                row,col = int_to_row_column(white_capture_squares[i-1])
                board[pos_opp_capture,row,col] += 1

        #fill in last taken and requested moves
        if i > 0:

            requested_move = black_requested_moves[i-1]['value'] if black_requested_moves[i-1] else None
            if requested_move:
                loc = move_to_location(requested_move,own_pieces)
                board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        #fill in whether last move captured a piece
        if i > 0:
            if black_capture_squares[i-1] is not None:
                row,col = int_to_row_column(black_capture_squares[i-1])
                board[pos_last_move_captured,row,col] += 1

        #fill in whether last move returned None
        if i > 0:
            if black_taken_moves[i-1] is None:
                board[pos_last_move_None,:,:] = 1
        
        #fill in own board
        own_pieces = chess.Board(black_fens_before[i])
        target_pieces = torch.zeros(6)
        for square,piece in own_pieces.piece_map().items():
            if not piece.color:
                row,col = int_to_row_column(square)
                board[pos_own_pieces+piece_map[str(piece)][1],row,col]+= 1
            else:
                target_pieces[piece.piece_type-1] += 1
        
        #add sense training data
        black_input_history.append(board)    

        
        target_sense = torch.zeros(8,8)
        if sense is not None:
            sense_row,sense_col = int_to_row_column(sense)
            target_sense[sense_row,sense_col] += 1  
            black_presense_observations.append(torch.clone(torch.stack(black_input_history[-20:])))
        
        #fill in sensed area     
        if sense is not None:           
            board[pos_sensed] += sense_location_to_area(sense)

        #fill in sensed pieces
        for res in result:
            if res[1] is not None:
                c,pos = piece_map[res[1]['value']]
                if c:
                    row,column = int_to_row_column(res[0])
                    board[pos_sense_result+pos,row,column] += 1
        
        tmp_move = torch.zeros(73,8,8)
        done_move = black_requested_moves[i]['value'] if black_requested_moves[i] else None
        target_move = torch.zeros(73*8*8+1)
        if done_move:
            if done_move[-1] == 'q':
                done_move = done_move[:-1]
            loc = move_to_location(done_move,own_pieces)
            tmp_move[loc[0],loc[1],loc[2]] = 1     
            target_move[:-1] = tmp_move.view(-1)
        else:
            continue
        black_input_history[-1] = board
        black_premove_observations.append(torch.clone(torch.stack(black_input_history[-20:])))
    return white_presense_observations,white_premove_observations,black_presense_observations,black_premove_observations

    

def json_to_dataset(processing_fn,check_if_exists = False):
    path = '/home/fawler/tbertram/RBC/data/games/'
    possible_opponents = ['Strangefish2','Fianchetto','Strangefish','Stockenstein','penumbra','LaSalle Bot', 'Oracle','ChessProfessorX',
    'DynamicEntropy','Kevin','trout','attacker']
    pool = mp.Pool(processes = 500)
    files = reversed(os.listdir(path))
    [pool.apply(processing_fn,args = (file,possible_opponents)) for file in files]

row_to_int = {7:'1',6:'2',5:'3',4:'4',3:'5',2:'6',1:'7',0:'8'}
row_to_int_reverse = {int(value):key for (key,value) in row_to_int.items()}
col_to_int = {7:'h',6:'g',5:'f',4:'e',3:'d',2:'c',1:'b',0:'a'}
pos_opp_capture = 0
pos_last_moves = 1
pos_last_move_captured = 74
pos_last_move_None = 75
pos_own_pieces = 76
pos_sensed = 82
pos_sense_result = 83
col_to_int_reverse = {value:key for (key,value) in col_to_int.items()}
codes = {}
borders = torch.zeros(73, 8,8, dtype= torch.bool)
codes, borders = fill_codes(codes,borders)
inverse_codes = {value:key for (key,value) in codes.items()}
columns = { k:v for v,k in enumerate("abcdefgh")}
piece_map = {'p': (0,0),'r': (0,1),'n': (0,2),'b': (0,3),'q': (0,4),'k': (0,5),'P': (1,0),
    'R': (1,1),'N': (1,2),'B': (1,3),'Q': (1,4),'K': (1,5)}
notation_map = {'a':0, 'b':1,'c':2, 'd':3,'e':4, 'f':5,'g':6, 'h':7,
    '1':7,'2':6,'3':5,'4':4,'5':3,'6':2,'7':1,'8':0,
    1:7,2:6,3:5,4:4,5:3,6:2,7:1,8:0}
    
if __name__ == '__main__': 
    json_to_dataset()