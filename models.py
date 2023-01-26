from torch.utils.data import Dataset, DataLoader
import torch
import os
import lzma
import pytorch_lightning as pl
import torch.nn.functional as F
import sqlite3
from sklearn.manifold import TSNE
import pickle
import random
from random import shuffle
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import torchmetrics
import time
import torch.optim as optim
import numpy as np
import torch.nn as nn
import chess


piece_to_index= {
    'P':0, 'R':1,'N':2,'B':3,'Q':4,'K':5,
    'p':6, 'r':7,'n':8, 'b': 9,'q':10,'k':11
}

def collate_fn(batch):
    positions = []
    results = []

    for batch_index in range(len(batch)):
        if not batch[batch_index]:
            continue

        board = board_from_fen(batch[batch_index][0])
        turn = batch[batch_index][1]
        result = batch[batch_index][2]
        positions.append(torch.cat((board,turn),0))
        results.append(result)

    positions = torch.stack(positions)
    results = torch.stack(results)
    assert positions.size(0) == results.size(0)
    return positions,results

class Supervised_Dataset(Dataset):
    def __init__(self,folder):
        self.folder = folder
        self.files = os.listdir(self.folder)
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file = torch.load(self.folder+self.files[index])
        return file[0],file[1],file[2],file[3]

class Stockfish_Dataset(Dataset):
    def __init__(self):
        super(Stockfish_Dataset,self).__init__()
        self.path = 'data/stock.db'

        self.con = sqlite3.connect(self.path)
        # creating cursor
        self.cur = self.con.cursor()
        # reading all table names
        self.fens = [a[0] for a in self.cur.execute("SELECT fen FROM evaluations")]
        self.evals = [a[0] for a in self.cur.execute("SELECT eval FROM evaluations")]


    def __getitem__(self, index):
        return self.fens[index], self.evals[index]

    def __len__(self):
        return len(self.fens)

def collate_fn(batch):
    import utils
    positions = []
    results = []

    for batch_index in range(len(batch)):
        if not batch[batch_index]:
            continue

        board = utils.board_from_fen(batch[batch_index][0])
        capture_square = batch[batch_index][1]
        capture_layer = torch.zeros((1,8,8))
        if capture_square:
            row, col = utils.int_to_row_column(capture_square)
            capture_layer[0,row,col] = 1
        turn = batch[batch_index][2]
        result = batch[batch_index][3]
        positions.append(torch.cat((board,capture_layer,turn),0))
        results.append(result)

    positions = torch.stack(positions)
    results = torch.stack(results)
    assert positions.size(0) == results.size(0)
    return positions,results

def collate_fn_stockfish(batch):
    
    import utils
    positions = []
    results = []

    for batch_index in range(len(batch)):
        if not batch[batch_index]:
            continue

        fen = batch[batch_index][0]
        eval = torch.Tensor([ 1/(1+10**(-batch[batch_index][1]/4))])
        board = utils.board_from_fen(fen)
        if 'w' in fen:
            turn = torch.ones((1,8,8))
        else:
            turn = torch.zeros((1,8,8))
        positions.append(torch.cat((board,torch.zeros((1,8,8)),turn),0))
        results.append(eval)

    positions = torch.stack(positions)
    results = torch.stack(results)
    assert positions.size(0) == results.size(0)
    return positions,results

class Evaluation_Block(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,size = 3):
        super(Evaluation_Block,self).__init__()
        self.input = nn.Conv2d(input_size,hidden_size, kernel_size=size, padding = 'same')
        self.norm = nn.BatchNorm2d(hidden_size)
        self.activation = nn.ELU()
        self.output = nn.Conv2d(hidden_size, output_size, kernel_size=size, padding = 'same')
        self.norm_2 = nn.BatchNorm2d(output_size)
        self.activation_2 = nn.ELU()

    def forward(self,x):
        inp = x
        x = self.input(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.norm_2(x)
        x = x+inp
        x = self.activation_2(x)
        return x

class Evaluation_Network(pl.LightningModule):   
    def __init__(self,stockfish = False, path = None,num_first_blocks = 5,num_second_blocks = 5, batch_size = 1024, first_block_size = 40, second_block_size = 100):
        self.output_layer_num = second_block_size
        self.path = path
        self.stockfish = stockfish
        if self.stockfish:
            stock_data = Stockfish_Dataset()
            train_size = int(len(stock_data)*0.9)
            self.train_set,self.val_set = torch.utils.data.random_split(stock_data,[train_size,len(stock_data)-train_size])
        super(Evaluation_Network,self).__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(14,first_block_size,kernel_size=5, padding = 'same'),
            nn.BatchNorm2d(first_block_size),
            nn.ELU()
        )
        self.first_hidden_list = nn.ModuleList()
        for i in range(num_first_blocks):
            self.first_hidden_list.append(
                Evaluation_Block(first_block_size,first_block_size,first_block_size,size = 5)
            )
        self.intermediate_block = nn.Sequential(
            nn.Conv2d(first_block_size,second_block_size,kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(second_block_size),
            nn.ELU()
        )
        self.second_hidden_list = nn.ModuleList()
        for i in range(num_second_blocks):
            self.second_hidden_list.append(
                Evaluation_Block(second_block_size,second_block_size,second_block_size,size = 3)
            )
        self.last_conv_block = nn.Sequential(
            nn.Conv2d(second_block_size,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ELU()
        )
        self.output_block = nn.Sequential(
            nn.Linear(1*8*8,256),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(256,256),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(256,256),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(256,1)
        )

        if self.stockfish:
            self.loss_fn = torch.nn.functional.mse_loss
        else:
            self.loss_fn = nn.functional.binary_cross_entropy_with_logits
        self.train_accuracy = torchmetrics.MeanSquaredError()
        self.validation_accuracy = torchmetrics.MeanAbsoluteError()
        self.test_accuracy = torchmetrics.MeanAbsoluteError()
        self.batch_size = batch_size
        self.num_workers = 32
    
    def forward(self,x):
        pre_linear = self.input_block(x)
        for block in self.first_hidden_list:
            pre_linear = block(pre_linear)
        pre_linear = self.intermediate_block(pre_linear)
        for block in self.second_hidden_list:
            pre_linear = block(pre_linear)
        pre_linear = self.last_conv_block(pre_linear)
        pre_linear = pre_linear.view(-1,32*8*8) 
        pre_linear = self.output_block(pre_linear)
        return pre_linear
    
    def training_step(self,batch,batch_idx):
        x,y = batch
        x = self(x)
        if self.stockfish:
            x = torch.sigmoid(x)
        loss = self.loss_fn(x,y)
        if batch_idx%100 == 0:
            if self.stockfish:
                print(batch_idx,x,y, loss.item())
            else:
                print(batch_idx,torch.sigmoid(x),y, loss.item())
        self.log('train_loss',loss.item(), sync_dist = True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        x = torch.sigmoid(self(x))
        loss = self.validation_accuracy(x,y)
        self.validation_accuracy.update(x,y)
        self.log('validation_loss',loss.item(), sync_dist = True)
        return loss

    
    def test_step(self,batch,batch_idx):
        x,y = batch
        x = self(x)
        loss = self.test_accuracy(x,y)
        self.test_accuracy.update(x,y)
        self.log('test_loss',loss.item(), sync_dist = True)
        return loss
            
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 3e-4)
        return optimizer

        
    def train_dataloader(self):
        if self.stockfish:
            return DataLoader(self.train_set,batch_size = self.batch_size,shuffle = True, pin_memory = True, collate_fn= collate_fn_stockfish, num_workers=self.num_workers, persistent_workers=False)
        else:
            train_data = Supervised_Dataset(f'{self.path}/train/')
            return DataLoader(train_data,batch_size = self.batch_size,shuffle = True, pin_memory = True, collate_fn= collate_fn, num_workers=self.num_workers, persistent_workers=False)

    def val_dataloader(self):
        if self.stockfish:
            return DataLoader(self.val_set,batch_size = self.batch_size,shuffle = False, pin_memory = True, collate_fn= collate_fn_stockfish, num_workers=self.num_workers, persistent_workers=False)
        else:
            val_data = Supervised_Dataset(f'{self.path}/val/')
            return DataLoader(val_data,batch_size = self.batch_size,shuffle = False, pin_memory = True, collate_fn= collate_fn, num_workers=self.num_workers, persistent_workers=False)

    def test_dataloader(self):
        test_data = Supervised_Dataset(f'{self.path}test/')


class Linear_Block(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,size = 3):
        super(Linear_Block,self).__init__()
        self.input = nn.Linear(input_size,hidden_size)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.dropout_1 = nn.Dropout()
        self.activation = nn.ELU()
        self.output = nn.Linear(hidden_size, output_size)
        self.norm_2 = nn.BatchNorm1d(output_size)
        self.dropout_2 = nn.Dropout()
        self.activation_2 = nn.ELU()

    def forward(self,x):
        inp = x
        x = self.input(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.output(x)
        x = self.norm_2(x)
        x = self.dropout_2(x)
        x = x+inp
        x = self.activation_2(x)
        return x


def replace_output_block(network):

    network.output_block= nn.Sequential(
            Linear_Block(32*8*8,2048,2048),
            nn.Linear(2048,512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ELU(),
            Linear_Block(512,512,512),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.ELU(),
            Linear_Block(256,256,256),
            nn.Linear(256,1)
        )
    network.last_conv_block = nn.Sequential(
            nn.Conv2d(256,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

class Siamese_RBC_dataset(Dataset):
    def __init__(self, path,num_choices,max_samples = None):
        self.path = path
        self.files = []
        for i,file in enumerate(sorted(os.scandir(path+'move/'), key = lambda k: random.random())):
            self.files.append(file.name)
            if max_samples and i == (max_samples//2)-1:
                break
        self.num_move_files = i+1
        for i,file in enumerate(sorted(os.scandir(path+'sense/'), key = lambda k: random.random())):
            self.files.append(file.name)
            if max_samples and i == (max_samples//2)-1:
                break
        self.num_sense_files = i+1
        self.num_choices = num_choices
        self.max_samples = max_samples
        self.num_players, self.player_encoding, self.empty_encoding = Siamese_RBC_dataset.create_player_encoding()
    
    def __len__(self):
        return self.num_move_files+self.num_sense_files

    def __getitem__(self,id):
        if id >= self.num_move_files:
            path = self.path+'sense/'
        else:
            path = self.path+'move/'
        try:
            with lzma.open(path+self.files[id], 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(id)
            print(len(self))
            print(e)
            anchor,positive,negatives,lengths,enc,empty_enc = self.__getitem__(id+1)
            return anchor,positive,negatives,lengths,enc,empty_enc

        else:
            anchor = data[0]
            positive = board_from_fen(data[1])
            if data[2]:
                if self.num_choices:
                    shuffle(data[2])
                    negatives = []
                    for b in data[2]:
                        b_tensor = board_from_fen(b)
                        if not torch.equal(positive,b_tensor):
                            negatives.append(b_tensor)
                        if len(negatives) >= self.num_choices:
                            break
                else:
                    negatives = []
                    for b in data[2]:
                        b_tensor = board_from_fen(b)
                        if not torch.equal(positive,b_tensor):
                            negatives.append(b_tensor)
                
                if len(negatives) > 0:
                    if data[3] in self.player_encoding:
                        return anchor,positive,negatives,len(negatives),self.player_encoding[data[3]],self.empty_encoding
                    else:
                        return anchor,positive,negatives,len(negatives),self.empty_encoding, self.empty_encoding 
                else:
                    anchor,positive,negatives,lengths,enc,empty_enc = self.__getitem__(id+1)
                    return anchor,positive,negatives,lengths,enc,empty_enc
            else:
                anchor,positive,negatives,lengths,enc,empty_enc = self.__getitem__(id+1)
                return anchor,positive,negatives,lengths,enc,empty_enc


    @staticmethod
    def create_player_encoding(path = 'game_numbers.csv'):
        num_players = 50
        player_dict = {}
        player_encoding = {}
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f, delimiter = ',')
                for line in reader:
                    player_dict[line[0]] = int(line[1])
        except:
            with open('game_numbers.csv', 'r') as f:
                reader = csv.reader(f, delimiter = ',')
                for line in reader:
                    player_dict[line[0]] = int(line[1])
        
        player_tuples = sorted(player_dict.items(), key = lambda x: x[1],reverse = True)
        valid_players = [p[0] for p in player_tuples[:num_players]]
        empty_encoding = torch.zeros((num_players,8,8))
        for i,p in enumerate(valid_players):
            encoding = torch.zeros_like(empty_encoding)
            encoding[i,:,:] = 1
            player_encoding[p] = encoding
        return num_players,player_encoding,empty_encoding

    
def board_from_fen(board):
    tensor_board = torch.zeros(12,8,8)
    board = chess.Board(board)
    for square,piece in board.piece_map().items():
        tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
    return tensor_board



class Embedding_Network(nn.Module):
    def __init__(self,input_layers,output_size):
        super(Embedding_Network,self).__init__()

        self.input = nn.Conv2d(input_layers,20,kernel_size=3, padding = 'same')
        self.embedding = nn.Linear(20*8*8,output_size)

    def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        x = x.view(-1, 20*8*8)
        x = self.embedding(x)
        #x = torch.tanh(x)

        return x

class Embedding_Transformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.seq_len = 20
        #BxCx8x8 -> B*20x90x8x8
        self.input_obs = nn.Conv2d(90,4,kernel_size=3, padding = 'same')
        self.fcc_obs = nn.Linear(4*8*8,256)
        #Bx50x8x8 -> B*20x50x8x8
        self.input_player = nn.Conv2d(50,4,kernel_size=3, padding = 'same')
        self.fcc_pl = nn.Linear(4*8*8,32)

        self.pos_encoding = nn.Embedding(self.seq_len, 288)


        encoder_layer = nn.TransformerEncoderLayer(d_model = 288, nhead = 8, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = 3)
        self.output = nn.Linear(288,512)

    def forward(self,obs,player):
        #BxCx8x8
        B,C,row,col = obs.shape
        obs = obs.view(B*self.seq_len,90,8,8)
        obs = self.input_obs(obs) #B*20,4,8,8
        obs = obs.view(B*self.seq_len,4*8*8)
        obs = self.fcc_obs(obs) #B*20,256
        obs = obs.view(B,self.seq_len,256)
        player = player.view(B,1,50,8,8)
        player = player.repeat(1,self.seq_len,1,1,1)
        player = player.view(B*self.seq_len,50,8,8)
        player = self.input_player(player) #B*20,4,8,8
        player = player.view(B*20,4*8*8)
        player = self.fcc_pl(player) #B*20,32
        player = player.view(B,self.seq_len,32)

        full_embedding = torch.cat((obs,player),dim = 2) #B,20,288
        pos_embedding = self.pos_encoding(torch.arange(self.seq_len, device = full_embedding.device))
        output = self.encoder(full_embedding+pos_embedding)
        output = output.mean(1) #B, 288
        output = self.output(output) #B, 512
        return output

class Player_Embedding_Network(nn.Module):
    def __init__(self,input_size):
        super(Player_Embedding_Network,self).__init__()

        self.input = nn.Linear(input_size, 256)
        self.embedding = nn.Linear(256,128)

    def forward(self,x):
        x = self.input(x)
        x = F.relu(x)
        x = self.embedding(x)
        x = torch.tanh(x)

        return x

class Siamese_Block(nn.Module):
    def __init__(self,in_size,out_size):
        super(Siamese_Block,self).__init__()
        self.linear = nn.Linear(in_size,out_size)
        self.dropout = nn.Dropout()
        self.activation = nn.ELU()
    
    def forward(self,x):
        return self.activation(self.dropout(self.linear(x)))

class Siamese_Network(pl.LightningModule):
    def __init__(self,test_choices,embedding_dimensions,use_weighting, player_distance_weight = 0, path = None, pre_embed_dim = 512):
        super(Siamese_Network,self).__init__()
        
        self.input_board = Embedding_Network(12,pre_embed_dim)
        self.input_observations = Embedding_Network(50+1800,pre_embed_dim)
        num_players,self.player_encoding,self.empty_encoding = Siamese_RBC_dataset.create_player_encoding('strangefish/game_numbers.csv')
        self.reverse_encoding = {torch.argmax(v,dim=0)[0,0].item():k for k,v in self.player_encoding.items()}
        self.test_choices = test_choices
        self.player_distance_weight = player_distance_weight
        self.embedding_dimensions = embedding_dimensions
        self.use_weighting = use_weighting
        self.train_pick_choices = []
        self.visualize = False
        main_block_size = 1024

        
        self.first_block = Siamese_Block(pre_embed_dim,main_block_size)


        self.main_network = nn.Sequential(
            nn.Linear(pre_embed_dim, 1024),
            nn.Dropout(),
            #nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            #nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(1024,1024),
            #nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(1024,embedding_dimensions),
            nn.Tanh()
        )
        self.main_network = nn.ModuleList()
        for _ in range(5):
            self.main_network.append(Siamese_Block(1024,1024))

        self.last_block = nn.Sequential(
            nn.Linear(1024,embedding_dimensions),
            nn.Tanh()
        )

        self.loss_fn = nn.TripletMarginLoss(reduction = 'none')
        self.train_accuracy = torchmetrics.Accuracy()
        self.accuracy = torchmetrics.Accuracy()
        self.train_pick_accuracy = []
        self.pick_accuracy = []
        self.pick_choices = []
        self.train_losses = []
        self.eval_losses = []
        self.test_losses = []
        self.pick_distance = []
        self.train_pick_distance = []
        self.choice_num_to_accuracy = defaultdict(list)
        self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
        self.seen_obs = []
        self.seen_players = []

        if path is not None:
            self.load_state_dict(torch.load(path))
            self.eval()
    
    def forward(self,anchor,positive,negative, player = None):
        anchor_out = self.anchor_forward(anchor,player)
        positive_out = self.choice_forward(positive)
        negative_out = self.choice_forward(negative)
        # if self.multianchor:
        #     assert player != None
        #     player_out = self.player_forward(player)
        #     return anchor_out,positive_out,negative_out,player_out
        return anchor_out,positive_out,negative_out

    def anchor_forward(self,anchor,player):
        full_anchor = torch.cat((anchor,player.view(-1,50,8,8)),dim = 1)
        anchor = self.main_network(self.input_observations(full_anchor))
        return anchor

    def choice_forward(self,choice):
        choice = self.main_network(self.input_board(choice))
        return choice

    def player_forward(self,player):
        player = self.main_network(self.input_player(player))
        return player




    
    def training_step(self,batch,batch_idx):
        anchor,positive,negative,lens,player = batch
        #print(player[0],anchor[0],positive[0],negative[0])
        anchor_out,positive_out,negative_out = self(anchor,positive,negative,player)
        #print(anchor_out[0],positive_out[0],negative_out[0])
        unchanged_loss = self.loss_fn(anchor_out,positive_out,negative_out)
        if torch.isnan(unchanged_loss).any():
            nan_index = torch.argmax(unchanged_loss.float()).item()
            print(anchor[nan_index],player[nan_index],positive[nan_index],negative[nan_index])
            print(anchor_out[nan_index],positive_out[nan_index],negative_out[nan_index])
            print(unchanged_loss[nan_index])
        #print(unchanged_loss)
        if batch_idx % 1000 == 0:
            torch.save(self.state_dict(),'siamese_players_wip.pt')
            # if batch_idx == 0:
            #     print(anchor[0,:,:,:].shape)
            #     print(player[0,:].shape)
            #     print(f'Embedded anchor: {anchor_out}')
            #     print(f'Pre_embedded anchor: {self.anchor_forward(anchor)}')
            #     print(f'Obs loss: {obs_loss, torch.mean(obs_loss)}')
            #     print(f'Embedded Player: {player_out}')
            #     print(f'Pre_embedded player: {self.player_forward(player)}')
            #     print(f'Player loss: {pl_loss, torch.mean(pl_loss)}')
        if self.use_weighting:
            loss = unchanged_loss*lens
        else:
            loss = unchanged_loss
        loss = torch.mean(loss)
        self.train_losses.append(loss.item())
        self.log('train_loss',loss)

        
        # def get_contributing_params(y, top_level=True):
        #     nf = y.grad_fn.next_functions if top_level else y.next_functions
        #     for f, _ in nf:
        #         try:
        #             yield f.variable
        #         except AttributeError:
        #             pass  # node has no tensor
        #         if f is not None:
        #             yield from get_contributing_params(f, top_level=False)


        # contributing_parameters = set(get_contributing_params(positive_out))
        # all_parameters = set(self.parameters())
        # non_contributing_pos = all_parameters - contributing_parameters
        # print(f'Pos: {non_contributing_pos}')  # returns the [999999.0] tensor
        # contributing_parameters = set(get_contributing_params(negative_out))
        # all_parameters = set(self.parameters())
        # non_contributing_neg = all_parameters - contributing_parameters
        # print(f'Neg: {non_contributing_neg}')  # returns the [999999.0] tensor
        # contributing_parameters = set(get_contributing_params(anchor_out))
        # all_parameters = set(self.parameters())
        # non_contributing_anch = all_parameters - contributing_parameters
        # print(f'Anch: {non_contributing_anch}')  # returns the [999999.0] tensor
        # contributing_parameters = set(get_contributing_params(player_out))
        # all_parameters = set(self.parameters())
        # non_contributing_pl = all_parameters - contributing_parameters
        # print(f'Pl: {non_contributing_pl}')  # returns the [999999.0] 
        # print(f'Never used: {set.intersection(non_contributing_anch,non_contributing_neg,non_contributing_pos,non_contributing_pl)}')

        # raise Exception


        return loss

    def validation_step(self,batch,batch_idx):
        anchor,positive,negative,lens,player = batch
        anchor_out,positive_out,negative_out = self(anchor,positive,negative,player)
        unchanged_loss = self.loss_fn(anchor_out,positive_out,negative_out)
        if self.use_weighting:
            loss = unchanged_loss*lens
        else:
            loss = unchanged_loss
        loss = torch.mean(loss)
        self.eval_losses.append(loss.item())
        self.log('eval_loss',loss)

        return loss

    def test_step(self,batch,batch_idx):
        if self.test_picks:
            anchor,choice_list,choice_lengths,players = batch
            maximum_length = torch.max(choice_lengths).item()
            print(choice_lengths)

            random_indizes = list(range(maximum_length))
            shuffle(random_indizes)
            try:
                correct_choice = np.argmin(random_indizes)
            except Exception as e:
                print(random_indizes,maximum_length, choice_lengths,choice_list.shape)
                raise e
            choice_list = choice_list.view(maximum_length,12,8,8)[random_indizes,:,:,:]
            batch_size = anchor.size(0)
        
            anchors_embedded = self.anchor_forward(anchor,players)
            choices = self.choice_forward(choice_list)
            choices = choices.view(batch_size,maximum_length,self.embedding_dimensions)
            obs_distances = torch.stack([get_distance(anchors_embedded,choices[:,j,:]).cpu() for j in range(choice_lengths)])
            obs_distances = obs_distances.transpose(0,1)
            distances = obs_distances

            for i in range(batch_size):
                if choice_lengths[i] < maximum_length:
                    distances[i,choice_lengths[i]:] = float('inf')
            picks = torch.argmin(distances,dim=1)
            pick_distances = torch.argsort(distances,descending = False,dim= 1)
            index_of_correct_choice = (pick_distances == correct_choice).nonzero(as_tuple=True)[1]
            self.pick_distance.extend(list(index_of_correct_choice))
            self.pick_accuracy.extend([1 if pick == correct_choice else 0 for pick in picks])
            for i in range(picks.shape[0]):
                if picks[i].item() == correct_choice:
                    self.choice_num_to_accuracy[choice_lengths[i].item()].append(1)
                    if torch.sum(players).item() == 64:
                        p = self.player_to_accuracy[torch.argmax(players,dim = 1)[0,0,0].item()]
                        p[choice_lengths[i].item()].append(1)
                else:
                    self.choice_num_to_accuracy[choice_lengths[i].item()].append(0)
                    if torch.sum(players).item() == 64:
                        p = self.player_to_accuracy[torch.argmax(players,dim = 1)[0,0,0].item()]
                        p[choice_lengths[i].item()].append(0)
            self.train_pick_choices.extend([choice_lengths[i].item() for i in range(choice_lengths.size(0))])


            if self.visualize:
                all_players = []
                empty_player = torch.zeros_like(players)
                embedded_empty = self.anchor_forward(anchor,empty_player)
                all_players.append(embedded_empty)
                for i in range(players.shape[1]):
                    new_player = torch.zeros_like(players)
                    new_player[0,i,:,:] = 1
                    if not torch.equal(new_player,players):
                        all_players.append(self.anchor_forward(anchor,new_player))
                num_players = len(all_players)
                all_players = torch.stack(all_players)
                all_things = torch.cat([anchors_embedded,all_players.squeeze(),choices.squeeze()], dim = 0)
                things_transformed = TSNE(n_components=2, learning_rate= 'auto').fit_transform(all_things.cpu().numpy())
                colors = ['g']+['r']+['black' for _ in range(num_players-1)]+['blue' for _ in range(choices.shape[1])]
                plt.scatter(things_transformed[:,0],things_transformed[:,1], c = colors, s = 10)
                plt.annotate('Pl',xy = things_transformed[0])
                plt.annotate('Empty',xy = things_transformed[1])
                plt.savefig('Siamese_vis.png')
                raise Exception
            return None
        else:
            anchor,positive,negative,lens = batch
            anchor_out,positive_out,negative_out = self(anchor,positive,negative)
            loss = self.loss_fn(anchor_out,positive_out,negative_out)
            if self.use_weighting:
                loss*=lens
            
            self.test_losses.extend([l.item() for l in loss])
            return None
            

    def training_epoch_end(self, outs) -> None:
        print(f'Seen observations : ')
        print(self.seen_obs)
        print(f'Seen players : ')
        print(self.seen_players)
        self.log('epoch_train_loss',np.mean(self.train_losses), sync_dist= True)
        torch.save(self.state_dict(),'siamese_players_wip.pt')
        self.train_losses.clear()

    def test_epoch_end(self,outs):
        if self.test_picks:
            accuracy = np.mean(self.pick_accuracy)
            self.pick_accuracy.clear()

            distance = np.mean(self.pick_distance)
            self.pick_distance.clear()

            test_choices = np.mean(self.train_pick_choices)
            self.train_pick_choices.clear()

            player_to_num_accuracy = self.player_to_accuracy.items()
            sorted_dict = {}
            for player,acc_dict in player_to_num_accuracy:
                num_tuples = acc_dict.items()
                num_tuples = sorted(num_tuples, key = lambda x: np.mean(x[1]))
                all_results = []
                for tup in num_tuples:
                    all_results.extend(tup[1])
                total_acc = np.mean(all_results)
                num_tuples = [(tup[0],np.mean(tup[1]),len(tup[1])) for tup in num_tuples]
                num_tuples.append(f'Total accuracy = {total_acc}')
                sorted_dict[self.reverse_encoding[player]] = num_tuples
                

            self.player_to_accuracy = defaultdict(lambda: defaultdict(list))
            self.epoch_accuracy = accuracy


            print(f'Pick accuracy: {self.epoch_accuracy}')
            print(f'Pick distance: {distance}')
            print(f'Pick choices: {test_choices}')
            print(f'Accuracy by player and choice number:')
            for k,v in sorted_dict.items():
                print(f'{k}:{v}')
            # self.log('pick_accuracy',self.epoch_accuracy)
            # self.log('pick_distance',distance)
            # self.log('choices',test_choices)
        else:
            print(self.test_losses[:100])
            print(np.mean(self.test_losses))
            self.log('test_loss',np.mean(self.test_losses), sync_dist= True)
            self.test_losses.clear()


    def validation_epoch_end(self,outs):
        epoch_loss =np.mean(self.eval_losses)
        self.log('epoch_eval_loss',epoch_loss, sync_dist = True)
        print(f'Epoch validation loss: {epoch_loss}')
        self.eval_losses.clear()
        # accuracy = np.mean(self.train_pick_accuracy)
        # self.train_pick_accuracy.clear()

        # distance = np.mean(self.train_pick_distance)
        # self.train_pick_distance.clear()
        # print(f'Train pick accuracy: {accuracy}')
        # print(f'Train pick distance: {distance}')
        # self.log('train_pick_accuracy',accuracy)
        # self.log('train_pick_distance',distance)
        torch.save(self.state_dict(),'siamese_players_wip.pt')

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 3e-4)
        return optimizer

class Siamese_Network_Tweaking(Siamese_Network):
    def __init__(self,test_choices,embedding_dimensions,use_weighting, player_distance_weight = 0, path = None):
        pre_embed_dim = 512
        super().__init__(test_choices,embedding_dimensions,use_weighting,player_distance_weight,path, pre_embed_dim = pre_embed_dim)
        device = super().device
        main_block_size = 1024

        self.input_observations = Embedding_Transformer(device = super().device)
        self.first_block = Siamese_Block(pre_embed_dim,main_block_size)
        self.main_network = nn.ModuleList()
        for _ in range(5):
            self.main_network.append(Siamese_Block(main_block_size,main_block_size))
        self.last_block = nn.Sequential(
            nn.Linear(main_block_size,embedding_dimensions)
        )
    
    def main_network_pass(self,input):
        input = self.first_block(input)
        for block in self.main_network:
            input = input + block(input)
        return self.last_block(input)

    def anchor_forward(self,anchor,player):
        anchor = self.main_network_pass(self.input_observations(anchor,player.view(-1,50,8,8)))
        return anchor

    def choice_forward(self,choice):
        choice = self.main_network_pass(self.input_board(choice))
        return choice

    def player_forward(self,player):
        player = self.main_network_pass(self.input_player(player))
        return player

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = 3e-4)
        return optimizer


def get_distance(positive,negative):
    return torch.sum(torch.pow(positive-negative,2),dim=1)

if __name__ == '__main__':
    train_path = 'data/siamese/train/'
    val_path = 'data/siamese/val/'
    train_data = Siamese_RBC_dataset(train_path,num_choices = 2,max_samples = 100)
    print(train_data[1])
    print(train_data[89])
    
    
