import chess.engine
import reconchess as rbc
import os
import chess.svg
import random
import numpy as np
from random import shuffle
from sklearn.preprocessing import minmax_scale
import utils as utils
import torch
import pandas as pd
from board_dict import BoardDict
import copy
from collections import defaultdict
import time
from board_statistics.custom_board import CustomBoard
from board_statistics.game_statistics import GameStatistics
from timer import Timer
import operator
from SiameseAgent import *

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'


class MultipleBoardBasic(Player):
    """
    Uses multiple boards and samples them randomly with stockfish to decide on moves
    Requires STOCKFISH_ENV_VAR set in order to work
    """

    def __init__(self):
        self.timer = Timer(900)
        self.board_dict = BoardDict()
        self.color = None
        self.first_turn = True
        self.sense_evaluation = None
        # True if you want more outputs
        self.debug = True
        # create the game statistics instance
        self.statistics = GameStatistics(game=self)
        self.turn_count = 0
        self.current_sense_result = list()
        self.taken_move_list = list()
        self.board_evaluations_sense = None
        self.save_csv = False
        self.times = {'Handle opponent move':[],'Choose sense': [], 'Choose move':[]}

        # make sure stockfish environment variable exists
        if STOCKFISH_ENV_VAR not in os.environ:
            os.environ['STOCKFISH_EXECUTABLE'] = "/home/fawler/tbertram/RBC/stockfish_14.1_linux_x64_ssse"
            

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
        self.siamese = SiameseAgent()

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        board = CustomBoard(board)  # we create an instance of our custom board
        self.board_dict.add_board(board)
        self.color = color
        self.track_turn = utils.Track_turn(self.color)
        if self.debug:
            print(chess.COLOR_NAMES[self.color])
        self.siamese.handle_game_start(color,board,opponent_name)
        self.opponent_name = opponent_name

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        start_time = time.time()
        len_boards_before = self.board_dict.size()

        current_boards = self.board_dict.get_boards()
        # for some reasons the game tries to notify of opponent turn on first move for white so ignore that
        if self.first_turn and self.color:
            self.first_turn = False
            self.times['Handle opponent move'].append(0)
            return
        else:
            self.siamese.handle_opponent_move_result(captured_my_piece,capture_square)
            resulting_boards = BoardDict()
            # if the opponent did not capture a piece, account for all possible moves on all boards
            if not captured_my_piece:
                
                # passing is an option so move old boards over
                for board in current_boards:
                    new_board = board.copy()
                    new_board.copy_custom_attributes(board)
                    new_board.push(chess.Move.null())
                    new_board.last_e_to_square = None  # we append None since we do not have a "last move"
                    resulting_boards.add_board(new_board)

                # save fens so that we don't have multiples of the same board
                fens = [self.reduce_fen(board.fen()) for board in resulting_boards.get_boards()]
                for board in current_boards:
                    # iterate through all moves:
                    for move in list(board.pseudo_legal_moves):
                        # exclude taking moves
                        if board.piece_at(move.to_square) is None and not board.is_en_passant(move):
                            new_board = board.copy()
                            new_board.copy_custom_attributes(board)
                            new_board.push(move)
                            new_fen = self.reduce_fen(new_board.fen())
                            if new_fen not in fens:
                                resulting_boards.add_board(new_board)
                                new_board.last_e_to_square = move.to_square
                                fens.append(new_fen)
                    # castling:
                    for board in utils.illegal_castling(board, self.color):
                        new_fen = self.reduce_fen(board.fen())
                        if new_fen not in fens:
                            resulting_boards.add_board(board)
                            fens.append(new_fen)
            # if a piece was captured, other moves have to be accounted for
            else:
                fens = []
                for board in current_boards:
                    # board.turn = not self.color
                    for move in board.pseudo_legal_moves:
                        # only look at the moves which captured on the given square
                        if move.to_square == capture_square:
                            new_board = board.copy()
                            new_board.copy_custom_attributes(board)
                            new_board.push(move)
                            new_fen = self.reduce_fen(new_board.fen())
                            if new_fen not in fens:
                                resulting_boards.add_board(new_board)
                                new_board.last_e_to_square = move.to_square
                                fens.append(new_fen)
                        #if the enemy can en passant 
                        if board.is_en_passant(move):
                            if self.color:
                                if not move.to_square+8 == capture_square:
                                    continue
                            else:
                                if not move.to_square - 8 == capture_square:
                                    continue
                            new_board = board.copy()
                            new_board.copy_custom_attributes(board)
                            new_board.push(move)
                            new_fen = self.reduce_fen(new_board.fen())
                            if new_board.piece_at(capture_square) is None and new_fen not in fens:
                                resulting_boards.add_board(new_board)
                                new_board.last_e_to_square = move.to_square
                                fens.append(new_fen)

            if self.debug:
                print("Opponent Move. Boards before handle_opponent_move_result: " + str(len_boards_before) + ", after: " + str(resulting_boards.size()) + "\n")
                if resulting_boards.size() < 1:
                    print('Not possible boards, this should not happen')
            overall_time = time.time() - start_time
            if self.save_csv:
                self.times['Handle opponent move'].append(overall_time)
            print(f'Handling opponent move result took {overall_time} seconds against {self.opponent_name}')
            if resulting_boards.size() == 0:
                self.backup_strategy = True
                print('WARNING DEFAULT TO BACKUP')
            self.board_dict = resulting_boards
            
    def square_to_row_column(self,square):
        return square//8,square%8

    def row_column_to_square(self,row,column):
        return row*8+column

    def get_adjacent_squares(self,square):
        adjacent_squares = []
        row,column = self.square_to_row_column(square)
        for i in range(-1,2):
            for j in range(-1,2):
                new_row = row+i
                new_column = column+j
                if 0 <= new_row < 8 and 0 <= new_column < 8:
                    adjacent_squares.append(self.row_column_to_square(new_row,new_column))
        return adjacent_squares

    def sense_result_to_string(self,board,squares,relevant_squares):
        result = ''
        for square in squares:
            if square in relevant_squares:
                res = board.piece_at(square)
                if res is not None:
                    result += str(res)
                else:
                    result += '0'
            else:
                result += 'x'
        return result

    def board_conflicts(self, boards):
        squares = np.zeros(64)
        relevant_squares = [index for index in range(64) if not boards[0].piece_at(index) or
                              boards[0].piece_at(index).color is not self.color]
        elimination_chances = []
        for square in range(64):
            sense_squares = self.get_adjacent_squares(square)
            square_results = defaultdict(int)
            elimination_chances_square = defaultdict(list)
            for board in boards:
                board_string = self.sense_result_to_string(board,sense_squares,relevant_squares)
                square_results[board_string] += 1
                elimination_chances_square[board_string].append(self.reduce_fen(board.fen()))
            total_weight = sum(square_results.values())
            elimination_chances_square = [(boards,square_results[res]/total_weight) for res,boards in elimination_chances_square.items()]
            elimination_chances.append(elimination_chances_square)
            if total_weight == 0:
                print('Why is total weight 0')
                total_weight = 1
            res = [(v/total_weight) * (total_weight-v) for v in square_results.values()]
            squares[square] = sum(res)
        return squares,elimination_chances

    def sense_weighted_reduction(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:

        self.timer.sync_timeout(seconds_left)
        self.statistics.update_statistics()
        start_time = time.time()
        sense_values,_ = self.board_conflicts(self.board_dict.get_boards())

        sense_values = sense_values.reshape((8,8))

        #Output            
        df_squares = pd.DataFrame(data = sense_values)
        if self.save_csv:
            df_squares.to_csv(f'debugging/sense_{len(self.siamese.board_list)+1}.csv', sep = ';', index = False)
        
        overall_time = time.time() - start_time
        if self.save_csv:
            self.times['Choose sense'].append(overall_time)
        print(f'Spent {overall_time} seconds on this sense against {self.opponent_name}')
        
        sense_values = sense_values.reshape(64)
        return sense_values

    def get_turn_number(self):
        return len(self.siamese.board_list) + 1

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:
        self.timer.sync_timeout(seconds_left)
        self.statistics.update_statistics()

        reduction_values = self.sense_weighted_reduction(sense_actions,move_actions,seconds_left)
        resulting_sense = np.argmax(reduction_values)
        self.siamese.last_sense = resulting_sense
        return sense_actions[resulting_sense]

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        len_boards_before = self.board_dict.size()

        # checks if our sensing was good
        if self.debug:
            print(f'Sense. Boards before handle_sense_result: {len_boards_before}, after:  {self.board_dict.size()} against {self.opponent_name}')
            if self.board_dict.size() < 1:
                print('Not possible boards, this should not happen')
     
    def reduce_fen(self,fen):
        return fen.split('-')[0]

    def get_passing_score(self,board,limit):
        own_king_square = board.king(self.color)
        # if there are any ally pieces that can take king, execute one of those moves
        own_king_attackers = board.attackers(not self.color, own_king_square)
        if own_king_attackers:
            return 0

        tmp_board = copy.deepcopy(board)
        tmp_board.turn = not self.color
        try:
            if self.color:
                score = self.engine.analyse(tmp_board,limit)['score'].white()
            else:
                score = self.engine.analyse(tmp_board,limit)['score'].black()
        except Exception as e:
            print(f'Something went wrong when passing. Exception {e}')
            print(tmp_board)
        numerical_score = score.score()
        if numerical_score is None:
            mate = score.mate()
            if mate > 0:
                win_chance = 1
            else:
                win_chance = 0
        else:
            numerical_score /= 100
            win_chance = 1/(1+10**(-numerical_score/4))
        
        return win_chance

    def get_best_move_per_board(self,board_list,limit):
        best_moves = []
        for board in board_list:
            enemy_king_square = board.king(not self.color)
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                move = chess.Move(attacker_square, enemy_king_square)
            else:

                try:
                    move = self.engine.play(board,limit).move
                except:
                    print('Something bad happened when choosing a move')
                    print(board)
                    move = chess.Move.null()
            if move is not None:
                best_moves.append(move)
        return best_moves

    def get_evaluations_of_board(self,board,move_options,limit):
        passing_score = self.get_passing_score(board,limit)
        board_voting = defaultdict(float)
        enemy_king_square = board.king(not self.color)
        # if there are any ally pieces that can take king, execute one of those moves
        enemy_king_attackers = board.attackers(self.color, enemy_king_square)
        if enemy_king_attackers:
            best_moves = [chess.Move(attack_square, enemy_king_square) for attack_square in enemy_king_attackers]
            for move in move_options:
                if move is None:
                    print('????????')
                if move in best_moves:
                    board_voting[move] = 1
                else:
                    if move.promotion is not None:
                        tmp_move = copy.deepcopy(move)
                        tmp_move.promotion = None
                        if tmp_move in best_moves:
                            board_voting[move] = 1
                            continue
                    if move in board.pseudo_legal_moves:
                        tmp_board = copy.deepcopy(board)
                        tmp_board.set_piece_at(move.to_square,tmp_board.piece_at(move.from_square))
                        tmp_board.remove_piece_at(move.from_square)
                        tmp_board.push(chess.Move.null())
                        tmp_board.clear_stack()
                        if tmp_board.attackers(not self.color, tmp_board.king(self.color)):
                            board_voting[move] = 0
                        else:
                            if tmp_board.is_stalemate():
                                if tmp_board.turn == self.color:
                                    board_voting[move] = 0
                                else:
                                    board_voting[move] = 1
                            else:
                                try:
                                    score = self.engine.analyse(tmp_board,limit)['score'].relative
                                except Exception as e:
                                    self.engine = chess.engine.SimpleEngine.popen_uci(os.environ[STOCKFISH_ENV_VAR], setpgrp=True)
                                    try:
                                        score = self.engine.analyse(tmp_board,limit)['score'].relative
                                    except:
                                        board_voting[move] = 0
                                        continue
                                numerical_score = score.score()
                                if numerical_score is None:
                                    mate = score.mate()
                                    if mate > 0:
                                        new_win_chance = 1
                                    else:
                                        new_win_chance = 0
                                else:
                                    numerical_score /= 100
                                    new_win_chance = 1/(1+10**(-numerical_score/4))
                                board_voting[move] = 1-new_win_chance
                    else:
                        board_voting[move] = passing_score
        else:
            for el in move_options:
                try:
                    if el not in board.legal_moves and el in board.pseudo_legal_moves:
                        new_win_chance = 0
                    else:
                        if el:
                            new_move = rbc.utilities.revise_move(board,el)
                        else:
                            new_move = el
                        if not new_move:
                            new_win_chance = passing_score
                        else:
                            tmp_board = copy.deepcopy(board)
                            tmp_board.set_piece_at(new_move.to_square,tmp_board.piece_at(new_move.from_square))
                            tmp_board.remove_piece_at(new_move.from_square)
                            tmp_board.push(chess.Move.null())
                            tmp_board.clear_stack()
                            if tmp_board.attackers(not self.color, tmp_board.king(self.color)):
                                new_win_chance = 0
                            elif tmp_board.is_stalemate():
                                if tmp_board.turn == self.color:
                                    board_voting[el] = 0
                                else:
                                    board_voting[el] = 1
                            else:
                                score = self.engine.analyse(tmp_board,limit)['score'].relative
                                numerical_score = score.score()
                                if numerical_score is None:
                                    mate = score.mate()
                                    if mate > 0:
                                        new_win_chance = 0
                                    else:
                                        new_win_chance = 1
                                else:
                                    numerical_score /= 100
                                    new_win_chance = 1-1/(1+10**(-numerical_score/4))
                    board_voting[el] = new_win_chance
                except Exception as e:
                    print('Something bad happened when choosing a move')
                    print(f'{str(e)} happened at board {board}')
                    print(score)
                    return None
        list_values = [board_voting[this_move] for this_move in move_options]
        if not list_values:
            print('???')
        return list_values
    
    def evaluate_moves(self,boards,move_options,columns, last_results = None):
        total_time_per_move = 1
        limit = chess.engine.Limit(time= min(total_time_per_move/(len(boards)*len(move_options)),0.1))
        all_values = defaultdict(lambda: defaultdict(list))
        min_boards = 10
        for i,board in enumerate(boards):
            if self.timer.remaining() < 30 and i >= min_boards or self.timer.remaining() < 180 and i >= 50 or self.timer.remaining() < 300 and i >= 100 or self.timer.elapsed_since_last() > 10 and i >= min_boards:
                i = i - 1
                break

            if last_results and self.reduce_fen(board.fen()) in last_results['weight'].keys():
                list_values = [last_results[move][self.reduce_fen(board.fen())] for move in move_options]
            else:
                list_values = self.get_evaluations_of_board(board,move_options,limit)
                if not list_values:
                    list_values = self.get_evaluations_of_board(board,move_options,limit)

            if self.reduce_fen(board.fen()) in columns.keys():
                old_series = columns[self.reduce_fen(board.fen())].tolist()
                columns[self.reduce_fen(board.fen())] = pd.Series(old_series[:-1]+['']+list_values+[1])
            else:
                columns[self.reduce_fen(board.fen())] = pd.Series(list_values + [1])
            for j,move in enumerate(move_options):
                all_values[self.reduce_fen(board.fen())][move] = list_values[j]
            all_values[self.reduce_fen(board.fen())]['weight'] = 1

        all_values_swapped = defaultdict(lambda:defaultdict(list))
        for key,value in all_values.items():
            for k,v in value.items():
                all_values_swapped[k][key] = v
        return all_values_swapped,i

    def get_evaluations_of_subset(self, all_values, boards, move):
        board_dict = all_values[move]
        worst_eval = 1
        worst_eval_weight = 0
        weighted_sum = 0
        total_weight = 0
        num_boards = 0
        for board in boards:
            eval = board_dict[board]
            weight = all_values['weight'][board]
            if eval < worst_eval:
                worst_eval = eval
                worst_eval_weight = weight
            worst_eval = min(worst_eval,eval)
            weighted_sum += weight * eval
            #weighted_sum += eval
            total_weight += weight
            num_boards += 1
        #weighted_sum /= sum(all_values['weight'].values())
       # weighted_sum /= num_boards
        if total_weight != 0:
            weighted_sum /= total_weight
        else:
            weighted_sum = board_dict[board]
        weighted_worst_eval = (worst_eval-1)*worst_eval_weight
        return (weighted_sum+weighted_worst_eval,weighted_sum,weighted_worst_eval)

    def choose_move_board_evaluation(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:# uses a list representation of our board_dict
        utils.equal_board_check(self.board_dict.get_boards(), self.color)
        start_time = time.time()
        boards = self.board_dict.get_boards()
        if self.board_evaluations_sense is not None:
            best_moves = list(self.board_evaluations_sense.keys())
            best_moves.remove('weight')
        else:
            best_moves = list(set(self.get_best_move_per_board(boards,chess.engine.Limit(time=0.001))))
        
        num_moves = len(best_moves)
        if num_moves == 0:
            print('We lost!')
            return np.random.choice(move_actions)

        if self.board_dict.size() == 0:
            print('Lost the real board')
            return None

        move_scores = {}
        columns = defaultdict(list)
        columns['Move options'] = pd.Series(best_moves + ['Weight of board'])
        all_values,_ = self.evaluate_moves(boards,best_moves, columns, last_results = self.board_evaluations_sense)

        for move in best_moves:
            move_scores[move] = self.get_evaluations_of_subset(all_values,all_values[move].keys(),move)

        try:
            sorted_votes = sorted(move_scores.items(), key= operator.itemgetter(1),reverse=True)
        except:
            print('????')
        df_output = pd.concat(columns,axis = 1)
        move_ranking = pd.Series([i[0] for i in sorted_votes])
        move_points = pd.Series([i[1] for i in sorted_votes])
        df_output['Best moves'] = pd.Series(move_ranking)
        df_output['Winrate of best moves'] = pd.Series(move_points)
        self.track_turn.check_turn_color(self.color)

        if self.save_csv:
            df_output.to_csv(f'debugging/distances_{len(self.siamese.board_list)+1}.csv', sep = ';', index = False)
        voted_move = sorted_votes[0][0]
        if voted_move != None and voted_move.promotion != None:
            voted_move.promotion = chess.QUEEN
        elif voted_move == None:
            print(f'Voted None-move against {self.opponent_name}')
        
        overall_time = time.time() - start_time
        if self.save_csv:
            self.times['Choose move'].append(overall_time)
        print(f'Spent {overall_time} seconds on this move against {self.opponent_name}')
        return voted_move

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        print(f'{seconds_left} seconds left against {self.opponent_name}')
        self.timer.sync_timeout(seconds_left)
        return self.choose_move_board_evaluation(move_actions,seconds_left)

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # uses a list representation of our board_dict
        utils.equal_board_check(self.board_dict.get_boards(), self.color)
        self.siamese.handle_move_result(requested_move,taken_move,captured_opponent_piece,capture_square)
        len_boards_before = self.board_dict.size()
        # handle result of own move, adapt possible board states
        resulting_boards = BoardDict()
        if taken_move is not None:
            if not captured_opponent_piece:
                print("Move did not capture a piece.")
                for board in self.board_dict.get_boards():
                    # if we did not capture a piece, we only keep boards
                    # where there was no piece on the square we moved to.
                    if board.piece_at(taken_move.to_square) is None and \
                            (board.is_pseudo_legal(taken_move) or board.is_castling(taken_move)):
                        new_board = board.copy()
                        new_board.copy_custom_attributes(board)
                        new_board.push(taken_move)
                        resulting_boards.add_board(new_board)
            else:
                print("Move captured a piece.")
                for board in self.board_dict.get_boards():
                    # if we captured a piece, we only keep boards
                    # where there was a opponent piece on the square we moved to.
                    if board.piece_at(capture_square) is not None and \
                            board.piece_at(capture_square).color is not self.color and \
                            board.is_pseudo_legal(taken_move) and \
                            capture_square is not board.king(not self.color):
                        new_board = board.copy()
                        new_board.copy_custom_attributes(board)
                        new_board.push(taken_move)
                        resulting_boards.add_board(new_board)

        # in the case the requested move was not possible
        elif requested_move != taken_move and taken_move is None:
            if self.debug:
                print("Move was rejected.")
            # if the actual move was different than the one we took (our move was not possible),
            # we only keep those boards where the requested move is not possible.
            for board in self.board_dict.get_boards():
                # we took a move, so its our turn for all boards
                new_board = board.copy()
                new_board.copy_custom_attributes(board)
                # new_board.turn = not self.color
                if not board.is_legal(requested_move):
                    new_board.push(chess.Move.null())
                    resulting_boards.add_board(new_board)
        # in the case we did not make a move
        else:
            for board in self.board_dict.get_boards():
                new_board = board.copy()
                new_board.copy_custom_attributes(board)
                new_board.push(chess.Move.null())
                resulting_boards.add_board(new_board)

        if self.debug:
            print(f"Own Move against {self.opponent_name}. Boards before handle_move_result: {len_boards_before}, after: {resulting_boards.size()}")
            if resulting_boards.size() < 1:
                print('Not possible boards, this should not happen')
        # Fixme
        # if we have 0 boards, then the king must have been captured. pls check
        # assert len(resulting_boards) >= 1 or capture_square is [board.king(not self.color) for board in self.board_dict.get_boards()]
        self.board_dict = resulting_boards

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason], game_history: GameHistory):
        try:
            del self.siamese
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
            torch.cuda.empty_cache()
        except chess.engine.EngineTerminatedError: 
            pass
