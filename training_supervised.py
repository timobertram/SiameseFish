import torch.nn as nn
from train_utils import *
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
from torch.utils.data import TensorDataset,DataLoader
from torch import optim
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning as pl
import models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import move_files
from argparse import ArgumentParser
import torch

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
        move_files.split_data(self.path)
        if self.stockfish:
            stock_data = models.Stockfish_Dataset()
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
        optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        return optimizer

        
    def train_dataloader(self):
        if self.stockfish:
            return DataLoader(self.train_set,batch_size = self.batch_size,shuffle = True, pin_memory = True, collate_fn= models.collate_fn_stockfish, num_workers=self.num_workers, persistent_workers=False)
        else:
            train_data = models.Supervised_Dataset(f'{self.path}/train/')
            return DataLoader(train_data,batch_size = self.batch_size,shuffle = True, pin_memory = True, collate_fn= models.collate_fn, num_workers=self.num_workers, persistent_workers=False)

    def val_dataloader(self):
        if self.stockfish:
            return DataLoader(self.val_set,batch_size = self.batch_size,shuffle = False, pin_memory = True, collate_fn= models.collate_fn_stockfish, num_workers=self.num_workers, persistent_workers=False)
        else:
            val_data = models.Supervised_Dataset(f'{self.path}/val/')
            return DataLoader(val_data,batch_size = self.batch_size,shuffle = False, pin_memory = True, collate_fn= models.collate_fn, num_workers=self.num_workers, persistent_workers=False)

    def test_dataloader(self):
        test_data = models.Supervised_Dataset(f'{self.path}test/')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", dest="path")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default = 4096, type = int)
    parser.add_argument("-n", "--network", dest = "pretrained_net")
    parser.add_argument("-s", "--stockfish", dest = "stockfish", default = False, type = bool)
    parser.add_argument("-f", "--filename", dest = "filename")

    parser = parser.parse_args()
    if not parser.path and not parser.stockfish:
        raise Exception('Specify path for data')
    if parser.path and parser.stockfish:
        raise Exception('Only set path or stockfish')
    network = Evaluation_Network(stockfish = parser.stockfish,num_first_blocks=0,num_second_blocks=40,first_block_size= 256, second_block_size=256, batch_size = parser.batch_size, path = parser.path)
    if parser.pretrained_net:
        print('Loading pretrained network')
        try:
            network.load_state_dict(torch.load(f'networks/{parser.pretrained_net}'))
        except:
            print('Replaced output')
            models.replace_output_block(network)
            network.load_state_dict(torch.load(f'networks/{parser.pretrained_net}'))
        print('Freezing feature extraction layers')
        for name, param in network.named_parameters():
            if not 'output_block' in name:
                param.requires_grad = False

        
        print('Resetting Batchnorm layer')
        for m in network.modules():
            if type(m) == nn.BatchNorm2d:
                m.reset_running_stats()
    else:
        print('Creating new network')


    early_stop_callback = EarlyStopping(monitor="validation_loss", min_delta=0.01, patience=3, verbose=True, mode="min")
    bar = TQDMProgressBar(refresh_rate = 1)
    
    trainer = pl.Trainer(auto_select_gpus = True, devices = 4, accelerator = 'gpu', \
     callbacks = [early_stop_callback,bar],max_epochs = -1, strategy=DDPStrategy(find_unused_parameters = False), profiler = 'simple',log_every_n_steps = 50)
    trainer.fit(network)
    name = parser.filename if parser.filename else 'rbc_network.pt'
    torch.save(network.state_dict(),name)
