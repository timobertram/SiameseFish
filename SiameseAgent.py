import models as models
from reconchess import *
import create_dataset as create_dataset
import copy
import utils as utils
import os
import numpy as np
import cProfile,pstats
import csv
import torch

pos_opp_capture = 0
pos_last_moves = 1
pos_last_move_captured = 74
pos_last_move_None = 75
pos_own_pieces = 76
pos_sensed = 82
pos_sense_result = 83
piece_map = {'p': (0,0),'r': (0,1),'n': (0,2),'b': (0,3),'q': (0,4),'k': (0,5),'P': (1,0),
    'R': (1,1),'N': (1,2),'B': (1,3),'Q': (1,4),'K': (1,5)}

class SiameseAgent():

    def __init__(self, device = None, player_embedding = False):
        self.network = models.Siamese_Network(None,256,False, multianchor= True, player_distance_weight= 0.03)
        self.player_embedding = player_embedding
        path = 'players/siamese_players.pt'
        try:
            self.network.load_state_dict(torch.load('networks/'+path))
        except:
            try:    
                self.network.load_state_dict(torch.load('scripts/networks/'+path))
            except:
                try:    
                    self.network.load_state_dict(torch.load('strangefish/networks/'+path))
                except Exception as e:
                    print('Cant load network')
                    print(os.curdir)
                    print(os.listdir(os.curdir))
                    raise e
            if not device:
                self.device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.network = self.network.to(self.device)
        self.network.eval()
        self.board_list = []
        self.current_board = torch.zeros(90,8,8)
        self.last_sense = None
        self.own_pieces = None
        self.color = None
        self.softmin = torch.nn.Softmin(dim=0)


    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        print('Siamese handled game start')
        self.color = color
        self.opponent_name = opponent_name
        if self.opponent_name in self.network.player_encoding.keys():
            print(f'Opponent {self.opponent_name} is in encodings')
        else:
            print(f'Opponent {self.opponent_name} is NOT in encodings')
        if color:
            self.current_board[-1,:,:] = 1
        self.own_pieces = copy.deepcopy(board)
        self.fill_own_pieces()

    def fill_own_pieces(self):
        for square,piece in self.own_pieces.piece_map().items():
            if piece.color == self.color:
                row,col = create_dataset.int_to_row_column(square)
                self.current_board[pos_own_pieces+piece_map[str(piece)][1],row,col] = 1
        

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        print('Siamese handled opponent move')
        if captured_my_piece:
            row,col = create_dataset.int_to_row_column(capture_square)
            self.current_board[pos_opp_capture,row,col] += 1
            self.own_pieces.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) \
            -> Optional[Square]:
        raise NotImplementedError

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        print('Siamese handled sense result')
        if self.last_sense is None:
            return
        
        if sense_result is not None:
            self.current_board[pos_sensed] += create_dataset.sense_location_to_area(self.last_sense)
            
            #fill in sensed pieces
            for res in sense_result:
                if res[1] is not None:
                    if str(res[1]) in piece_map:
                        c,pos = piece_map[str(res[1])]
                    else:
                        c,pos = piece_map[str(res[1]['value'])]
                    if c != self.color:
                        row,column = create_dataset.int_to_row_column(res[0])
                        self.current_board[pos_sense_result+pos,row,column] += 1

    def get_choice_embedding(self,choice):
        board = utils.board_from_fen(choice.fen())
        board = board.unsqueeze(0)
        return self.network.choice_forward(board.to(self.device))

    def get_player_embedding(self,player):
        if player in self.network.player_encoding.keys():
            embedded_player =self.network.player_forward(self.network.player_encoding[player].view(1,-1).to(self.device))
        else:
            embedded_player = self.network.player_forward(self.network.empty_encoding.view(1,-1).to(self.device))
        return embedded_player

    def get_board_weightings(self, possible_boards,profiled = False):
        if profiled:
            pr = cProfile.Profile()
            pr.enable()
        self.fill_own_pieces()
        num_options = len(possible_boards)
        if num_options == 0:
            print('No board options!')
            print(self.own_pieces.fen())
            return []

        #check for random board if own pieces align
        random_board = np.random.choice(list(possible_boards))
        for square,piece in random_board.piece_map().items():
            if piece.color == self.color:
                if self.own_pieces.piece_at(square) != piece:
                    print('Inconsistency between our board and the random board!')
                    print(f'Random board: {random_board.fen()}')
                    print(f'Own board: {self.own_pieces.fen()}')
                    raise Exception

        board_tensor = []
        board_list = list(possible_boards)
        for b in board_list:
            board_tensor.append(utils.board_from_fen(b.fen()))
            
        # with open(f'debugging/boards_{len(self.board_list)+1}.csv', 'w') as f:
        #     writer = csv.writer(f,delimiter = ';')
        #     for b in board_tensor:
        #         writer.writerow(b)


        board_tensor = torch.stack(board_tensor)
        board_tensors_embedded = self.network.choice_forward(board_tensor.to(self.device))

        history = self.get_history()
        # with open(f'debugging/history_{len(self.board_list)+1}.csv', 'w', newline='') as f:
        #     writer = csv.writer(f,delimiter = ';')
        #     writer.writerows(history)
        anchor = self.network.anchor_forward(history.to(self.device)).squeeze()
        anchor_repeated = anchor.repeat(num_options,1)
        
        distances = models.get_distance(anchor_repeated,board_tensors_embedded)


        #Get player embedding distances
        if self.player_embedding:
            embedded_player = self.get_player_embedding(self.opponent_name)
            player_distances = models.get_distance(embedded_player.squeeze().repeat(num_options,1),board_tensors_embedded)
            distances = distances + self.network.player_distance_weight*player_distances

        weights = self.distances_to_weights(distances)
        if (torch.isnan(torch.Tensor(weights)) == True).any():
            print(distances)
            print(anchor)
            print(board_list)
            print(board_tensors_embedded)
            print(weights)
            raise Exception


        if profiled:
            pr.disable()
            ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
        return weights,distances,anchor

    def get_updated_weights(self,anchor,old_distances,new_board):
        embedded_board = self.get_choice_embedding(new_board).squeeze()
        new_distance = models.get_distance(anchor.unsqueeze(dim=0),embedded_board.unsqueeze(dim=0))
        if self.player_embedding:
            pl_distance = models.get_distance(self.get_player_embedding(self.opponent_name),embedded_board)
            new_distance = new_distance + self.network.player_distance_weight * pl_distance
        all_distances = torch.cat((old_distances,new_distance),dim=0)
        return self.distances_to_weights(all_distances,ignore_last_for_min= True)


    def distances_to_weights(self,distances, ignore_last_for_min = False):
        if distances.size(0) > 1:
            distances += 1e-5
            if ignore_last_for_min:
                distances = torch.ones_like(distances)/ (distances/torch.min(distances[:-1]))
            else:
                distances = torch.ones_like(distances)/ (distances/torch.min(distances))
            #distances = ((distances-torch.min(distances))/(torch.max(distances)-torch.min(distances)))
            #distances = torch.softmax(distances,dim=0)
            #distances /= torch.sum(distances)
            #distances = self.softmin(distances)
        else:
            distances = torch.ones_like(distances)
        distances = distances.tolist()
        return distances


    def get_embeddings(self, possible_boards):
        self.fill_own_pieces()
        num_options = len(possible_boards)
        if num_options == 0:
            print('No board options!')
            print(self.own_pieces.fen())
            return []

        #check for random board if own pieces align
        random_board = np.random.choice(possible_boards)
        for square,piece in random_board.piece_map().items():
            if piece.color == self.color:
                if self.own_pieces.piece_at(square) != piece:
                    print('Inconsistency between our board and the random board!')
                    print(random_board.fen())
                    print(self.own_pieces.fen())

        board_tensor = []
        for b in possible_boards:
            board_tensor.append(utils.board_from_fen(b.fen()))
            
        # with open(f'debugging/boards_{len(self.board_list)+1}.csv', 'w') as f:
        #     writer = csv.writer(f,delimiter = ';')
        #     for b in board_tensor:
        #         writer.writerow(b)


        board_tensor = torch.stack(board_tensor)
        board_tensors_embedded = self.network.choice_forward(board_tensor.to(self.device))

        history = self.get_history()
        # with open(f'debugging/history_{len(self.board_list)+1}.csv', 'w', newline='') as f:
        #     writer = csv.writer(f,delimiter = ';')
        #     writer.writerows(history)
        anchor_embedded = self.network.anchor_forward(history.to(self.device)).squeeze()
        return anchor_embedded,board_tensors_embedded, possible_boards
    

    def get_history(self):
        history = torch.zeros(20,90,8,8)
        history[-1,:,:,:] = self.current_board
        for i in range(19):
            if len(self.board_list) > i:
                history[-2-i] = self.board_list[-1-i]
            else:
                break
        return history.view(1,20*90,8,8)


    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        
        self.board_list.append(self.current_board.clone())
        self.current_board = torch.zeros(90,8,8)
        if self.color:
            self.current_board[-1,:,:] = 1

        if requested_move is not None:
            requested_move = str(requested_move)
            if requested_move[-1] == 'q':
                promoted_to_queen = True
                requested_move = requested_move[:-1]
            else:
                promoted_to_queen = False
            loc = create_dataset.move_to_location(requested_move,self.own_pieces)
            self.current_board[pos_last_moves+loc[0],loc[1],loc[2]] += 1

        if captured_opponent_piece:
            row,col = create_dataset.int_to_row_column(capture_square)
            self.current_board[pos_last_move_captured,row,col] += 1
        
        
        if taken_move is None:
            self.current_board[pos_last_move_None,:,:] = 1
        else:
            if self.own_pieces.turn != self.color:
                self.own_pieces.push(chess.Move.null())

            if self.own_pieces.is_castling(taken_move):
                self.own_pieces.push(taken_move)
            else:
                piece = self.own_pieces.piece_at(taken_move.from_square)
                self.own_pieces.remove_piece_at(taken_move.from_square)
                if promoted_to_queen:
                    self.own_pieces.set_piece_at(taken_move.to_square,chess.Piece(chess.QUEEN,self.color))
                else:
                    self.own_pieces.set_piece_at(taken_move.to_square,piece)
            







