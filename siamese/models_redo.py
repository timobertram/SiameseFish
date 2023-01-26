from torch.utils.data import Dataset, DataLoader
import torch
import os
import lzma
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
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
class Siamese_RBC_dataset(Dataset):
    def __init__(self, path,num_choices,max_samples = None):
        self.path = path
        file_iterator = os.scandir(path)
        self.files = []
        for i,file in enumerate(file_iterator):
            self.files.append(file.name)
            if max_samples and i == max_samples:
                break
        self.num_choices = num_choices
        self.max_samples = max_samples
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self,id):
        try:
            with lzma.open(self.path+self.files[id], 'rb') as f:
                data = pickle.load(f)

            anchor = data[0]
            positive = self.board_from_fen(data[1])
            if data[2]:
                if self.num_choices:
                    negatives = [self.board_from_fen(b) for b in np.random.choice(data[2],self.num_choices)]
                else:
                    negatives = [self.board_from_fen(b) for b in data[2]]
                negatives = [n for n in negatives if not torch.equal(n,positive)]
                return anchor,positive,negatives,len(data[2])
            else:
                os.remove(self.path+self.files[id])
        except Exception as e:
            print(id)
            print(len(self))
            print(e)
            pass
        

    
    def board_from_fen(self,board):
        tensor_board = torch.zeros(12,8,8)
        board = chess.Board(board)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                tensor_board[piece_to_index[str(piece)],7-chess.square_rank(square),chess.square_file(square)] = 1
        return tensor_board



class Embedding_Network(nn.Module):
    def __init__(self,input_layers):
        super(Embedding_Network,self).__init__()

        self.input = nn.Conv2d(input_layers,20,kernel_size=3, padding = 'same')
        self.norm = nn.BatchNorm2d(20)
        self.embedding = nn.Linear(20*8*8,128)

    def forward(self,x):
        x = self.input(x)
        #x = self.norm(x)
        x = F.relu(x)
        x = x.view(-1, 20*8*8)
        x = self.embedding(x)
        x = F.relu(x)

        return x

class Siamese_Network(nn.Module):
    def __init__(self,test_choices,embedding_dimensions,use_weighting):
        super(Siamese_Network,self).__init__()
        self.input_board = Embedding_Network(12)
        self.input_observations = Embedding_Network(1800)
        self.test_choices = test_choices
        self.use_weighting = use_weighting
        self.test_picks = False

        self.main_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64,embedding_dimensions),
            nn.Tanh()
        )

        self.loss_fn = nn.TripletMarginLoss(reduction = 'mean')
        self.train_accuracy = torchmetrics.Accuracy()
        self.device = 'cuda'
        self.accuracy = torchmetrics.Accuracy()
        self.train_pick_accuracy = []
        self.pick_accuracy = []
        self.train_losses = []
        self.eval_losses = []
        self.test_losses = []
        self.pick_distance = []
        self.train_pick_distance = []
        self.configure_optimizers()
    
    def forward(self,anchor,positive,negative):
        anchor_out = self.anchor_forward(anchor.to(self.device))
        positive_out = self.choice_forward(positive.to(self.device))
        negative_out = self.choice_forward(negative.to(self.device))
        return anchor_out,positive_out,negative_out

    def anchor_forward(self,anchor):
        anchor = self.main_network(self.input_observations(anchor))
        return anchor

    def choice_forward(self,choice):
        choice = self.main_network(self.input_board(choice))
        return choice
    
    def training_step(self,batch,batch_idx):
        anchor,positive,negative,lens = batch
        anchor_out,positive_out,negative_out = self(anchor,positive,negative)
        unchanged_loss = self.loss_fn(anchor_out,positive_out,negative_out)
        if self.use_weighting:
            loss = unchanged_loss*lens
        else:
            loss = unchanged_loss
        loss = torch.mean(loss)
        # if loss.item() < 0.1:
        #     anchor_out = anchor_out.detach().cpu()
        #     positive_out = positive_out.detach().cpu()
        #     negative_out = negative_out.detach().cpu()
        #     plt.scatter([anchor_out[0,0]],[anchor_out[0,1]],label = 'anchor')
        #     plt.scatter([positive_out[0,0]],[positive_out[0,1]],label = 'positive')
        #     plt.scatter([negative_out[0,0]],[negative_out[0,1]],label = 'negative')
        #     plt.legend()
        #     plt.savefig(f'what_{time.time()}.png')
        #     plt.close()
        #     print(anchor[0],positive[0],negative[0])
        #     print(anchor_out[0],positive_out[0],negative_out[0])
        #     print(loss)
        #     anchor_out,negative_out,positive_out = self(anchor,negative,positive)
        #     print(anchor_out[0],negative_out[0],positive_out[0])
        #     reverse_loss = self.loss_fn(anchor_out,negative_out,positive_out)
        #     print(reverse_loss)
        #     print('finished')
        self.train_losses.append(loss.item())
        self.log('train_loss',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        anchor,positive,negative,lens = batch
        anchor_out,positive_out,negative_out = self(anchor,positive,negative)
        unchanged_loss = self.loss_fn(anchor_out,positive_out,negative_out)
        if self.use_weighting:
            loss = unchanged_loss*lens
        else:
            loss = unchanged_loss
        loss = torch.mean(loss)
        # if loss.item() < 0.1:
        #     anchor_out = anchor_out.detach().cpu()
        #     positive_out = positive_out.detach().cpu()
        #     negative_out = negative_out.detach().cpu()
        #     plt.scatter([anchor_out[0,0]],[anchor_out[0,1]],label = 'anchor')
        #     plt.scatter([positive_out[0,0]],[positive_out[0,1]],label = 'positive')
        #     plt.scatter([negative_out[0,0]],[negative_out[0,1]],label = 'negative')
        #     plt.legend()
        #     plt.savefig(f'what_{time.time()}.png')
        #     plt.close()
        #     print(anchor[0],positive[0],negative[0])
        #     print(anchor_out[0],positive_out[0],negative_out[0])
        #     print(loss)
        #     anchor_out,negative_out,positive_out = self(anchor,negative,positive)
        #     print(anchor_out[0],negative_out[0],positive_out[0])
        #     reverse_loss = self.loss_fn(anchor_out,negative_out,positive_out)
        #     print(reverse_loss)
        #     print('finished')
        self.eval_losses.append(loss.item())
        self.log('eval_loss',loss)
        return loss


        # anchor,choice_list = batch
        # batch_size = anchor.size(0)
        
        # anchors = self.anchor_forward(anchor)
        # choices = self.choice_forward(choice_list.view(batch_size*(self.test_choices+1),12,8,8))
        # choices = choices.view(batch_size,self.test_choices+1,-1)
        
        # distances = torch.stack([get_distance(anchors,choices[:,j,:]).cpu() for j in range(self.test_choices+1)])
        # picks = torch.argmin(distances,dim=0)
        # pick_distances = torch.argsort(distances,descending=False,dim= 0)
        # self.train_pick_distance.extend(list(torch.argmin(pick_distances,dim=0).numpy()))
        # self.train_pick_accuracy.extend([1 if pick == 0 else 0 for pick in picks])
        # return None

    
    def test_step(self,batch,batch_idx):
        if self.test_picks:
            anchor,choice_list = batch
            batch_size = anchor.size(0)
            
            anchors = self.anchor_forward(anchor)
            choices = self.choice_forward(choice_list.view(batch_size*(self.test_choices+1),12,8,8))
            choices = choices.view(batch_size,self.test_choices+1,-1)
            
            distances = torch.stack([get_distance(anchors,choices[:,j,:]).cpu() for j in range(self.test_choices+1)])
            picks = torch.argmin(distances,dim=0)
            pick_distances = torch.argsort(distances,descending = False,dim= 0)
            self.pick_distance.extend(list(torch.argmin(pick_distances,dim=0).numpy()))
            self.pick_accuracy.extend([1 if pick == 0 else 0 for pick in picks])
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
        self.log('epoch_train_loss',np.mean(self.train_losses))
        self.train_losses.clear()

    def test_epoch_end(self,outs):
        if self.test_picks:
            accuracy = np.mean(self.pick_accuracy)
            self.pick_accuracy.clear()

            distance = np.mean(self.pick_distance)
            self.pick_distance.clear()
            print(f'Pick accuracy: {accuracy}')
            print(f'Pick distance: {distance}')
            self.log('pick_accuracy',accuracy)
            self.log('pick_distance',distance)
        else:
            print(self.test_losses[:100])
            print(np.mean(self.test_losses))
            self.log('test_loss',np.mean(self.test_losses))
            self.test_losses.clear()


    def validation_epoch_end(self,outs):
        epoch_loss =np.mean(self.eval_losses)
        self.log('epoch_eval_loss',epoch_loss)
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

    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr = 3e-4)
        return self.optimizer


def get_distance(positive,negative):
    return torch.sum(torch.pow(positive-negative,2),dim=1)



if __name__ == '__main__':
    dataset = Siamese_RBC_dataset('dataset/')
    anchor,positive,negatives = dataset[2]
    
