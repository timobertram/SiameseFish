import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy,DDPSpawnStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset,DataLoader
from torch import optim, tensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import numpy as np
import cProfile, pstats, io
import models
from torch.utils.tensorboard import SummaryWriter
import torch

def collate_fn(batch):
    anchors = []
    positives = []
    negatives = []
    negative_lens = []

    for batch_index in range(len(batch)):
        if not batch[batch_index]:
            continue
        padded_anchor = torch.zeros(20,90,8,8)
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[-len_unpadded_anchor:,:,:,:] = batch[batch_index][0]
        padded_anchor = padded_anchor.view(20*90,8,8)
        pos = batch[batch_index][1]
        negative_list = batch[batch_index][2]
        for neg in negative_list:
            if torch.equal(pos,neg):
                continue
            anchors.append(padded_anchor)
            positives.append(pos)
            negatives.append(neg)
            negative_lens.append(torch.LongTensor([batch[batch_index][3]]))

    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    negative_lens = torch.stack(negative_lens).squeeze()
    assert anchors.size(0) == positives.size(0) == negatives.size(0) == negative_lens.size(0)
    return anchors,positives,negatives,negative_lens

def collate_fn_multianchor(batch):
    batch_size = sum([b[3] for b in batch if b is not None])
    anchors = torch.empty((batch_size,1800,8,8))
    positives = torch.empty((batch_size,12,8,8))
    negatives = torch.empty((batch_size,12,8,8))
    negative_lens = torch.empty((batch_size))
    encodings = torch.empty((batch_size,50,8,8))
    i = 0
    for batch_index in range(len(batch)):
        if batch[batch_index] is None or batch[batch_index][3] == 0:
            continue
        padded_anchor = torch.zeros((20,90,8,8))
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[-len_unpadded_anchor:,:,:,:] = batch[batch_index][0]
        padded_anchor = padded_anchor.view(20*90,8,8)
        pos = batch[batch_index][1]
        negative_list = batch[batch_index][2]
        negative_len = batch[batch_index][3]
        assert negative_len != 0
        player_encoding = batch[batch_index][4]
        for neg in negative_list:
            if torch.equal(pos,neg):
                print('WHAT IS THIS!!!!')
                print(pos,neg)
                continue
            anchors[i,:,:,:] = padded_anchor
            positives[i,:,:,:] = pos
            negatives[i,:,:,:] = neg
            negative_lens[i] = torch.LongTensor([negative_len])
            encodings[i,:,:,:] = player_encoding
        i += 1
        
        # if not torch.equal(empty_encoding,player_encoding):
        #     for neg in negative_list:
        #         if torch.equal(pos,neg):
        #             continue
        #         anchors.append(padded_anchor)
        #         positives.append(pos)
        #         negatives.append(neg)
        #         negative_lens.append(torch.LongTensor([batch[batch_index][3]]))
        #         encodings.append(player_encoding)
    # anchors = torch.stack(anchors)
    # positives = torch.stack(positives)
    # negatives = torch.stack(negatives)
    # negative_lens = torch.stack(negative_lens).squeeze()
    # encodings = torch.stack(encodings)
    if torch.isnan(anchors).any():
        print('Anchor')
    if torch.isnan(positives).any():
        print('Pos')
    if torch.isnan(negatives).any():
        print('Neg')
    if torch.isnan(encodings).any():
        print('Enc')
    return anchors,positives,negatives,negative_lens,encodings

def pick_collate_fn(batch):
    anchors = []
    choices = []
    choice_lengths = []

    for batch_index in range(len(batch)):
        if not batch[batch_index]:
            continue
        negative_list = batch[batch_index][2]
        pos = batch[batch_index][1]
        choice_list = [pos]
        padded_anchor = torch.zeros(20,90,8,8)
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[-len_unpadded_anchor:,:,:,:] = batch[batch_index][0]
        padded_anchor = padded_anchor.view(20*90,8,8)
        anchors.append(padded_anchor)
        for neg in negative_list:
            assert not torch.equal(pos,neg)
            choice_list.append(neg)
        while len(choice_list) < 11:
            choice_list.append(torch.zeros(12,8,8))
        choices.append(torch.stack(choice_list))
        choice_lengths.append(torch.LongTensor([len(negative_list)]))
    anchors = torch.stack(anchors)
    choices = torch.stack(choices)
    choice_lengths = torch.stack(choice_lengths)
    return anchors,choices,choice_lengths

def pick_collate_fn_multianchor(batch):
    batch_size = len(batch)
    max_length = max([b[3] for b in batch if b is not None])+1
    anchors = torch.empty((batch_size,1800,8,8))
    choices = torch.zeros((batch_size,max_length,12,8,8))
    choice_lengths = torch.empty((batch_size), dtype = torch.int)
    encodings = torch.empty((batch_size,50,8,8))
    i = 0
    for batch_index in range(len(batch)):
        if batch[batch_index] is None or batch[batch_index][3] == 0:
            continue    
        padded_anchor = torch.zeros((20,90,8,8))
        len_unpadded_anchor = batch[batch_index][0].size(0)
        padded_anchor[-len_unpadded_anchor:,:,:,:] = batch[batch_index][0]
        padded_anchor = padded_anchor.view(20*90,8,8)
        pos = batch[batch_index][1]
        negative_list = batch[batch_index][2]
        negative_len = batch[batch_index][3]
        player_encoding = batch[batch_index][4]

        choices[i,0,:,:,:] = pos
        anchors[i,:,:,:] = padded_anchor
        choice_lengths[i] = torch.LongTensor([negative_len+1])
        encodings[i,:,:,:] = player_encoding
        j = 1
        for neg in negative_list:
            if torch.equal(pos,neg):
                print('WHAT IS THIS!!!!')
                print(pos,neg)
                continue
            choices[i,j,:,:,:] = neg
            j += 1
        i += 1
    return anchors,choices,choice_lengths, encodings

def debugging_stuff():
    lengths = []
    for i,data in enumerate(train_data):
        if i == 100000:
            break
        if data:
            a,p,n = data
        else:
            print('Weird')
        lengths.append(len(n))

    plt.hist(lengths,density = True, bins = 100)
    plt.title(f'Average amount of negatives: {np.mean(lengths)}')
    plt.savefig('train_hist.png')
    plt.close()
    lengths = []
    for i,data in enumerate(test_data):
        if i == 100000:
            break
        if data:
            a,p,n = data
        else:
            print('Weird')
        lengths.append(len(n))
    plt.hist(lengths,density = True, bins = 100)
    plt.title(f'Average amount of negatives: {np.mean(lengths)}')
    plt.savefig('test_hist.png')
    plt.close()
    
    raise Exception
    pos = []
    negs = []
    for i,(a,p,n) in enumerate(train_data):
        if i == 100:
            with open('train_pos.csv','w') as f:
                writer = csv.writer(f,delimiter = ';')
                writer.writerows(pos)
            with open('train_neg.csv','w') as f:
                writer = csv.writer(f,delimiter = ';')
                writer.writerows(negs)
            break
        else:
            pos.append(p.numpy())
            negs.append(n[0].numpy())
    pos = []
    negs = []
    for i,(a,p,n) in enumerate(test_data):
        if i == 100:
            with open('test_pos.csv','w') as f:
                writer = csv.writer(f,delimiter = ';')
                writer.writerows(pos)
            with open('test_neg.csv','w') as f:
                writer = csv.writer(f,delimiter = ';')
                writer.writerows(negs)
            raise Exception
        else:
            pos.append(p.numpy())
            negs.append(n[0].numpy())

def more_debug():
    
    for a,p,n in train_loader:
        a = a[0].unsqueeze(dim=0)
        p = p[0].unsqueeze(dim=0)
        n = n[0].unsqueeze(dim=0)
        anchor_out,positive_out,negative_out = network(a,p,n)
        break
    
    anchor_out = anchor_out.detach().cpu()
    positive_out = positive_out.detach().cpu()
    negative_out = negative_out.detach().cpu()
    print(anchor_out,positive_out,negative_out)
    print(network.loss_fn(anchor_out,positive_out,negative_out))
    plt.scatter([anchor_out[0,0]],[anchor_out[0,1]],label = 'anchor')
    plt.scatter([positive_out[0,0]],[positive_out[0,1]],label = 'positive')
    plt.scatter([negative_out[0,0]],[negative_out[0,1]],label = 'negative')
    plt.legend()
    plt.savefig(f'what_train.png')

    
    for a,p,n in eval_loader:
        a = a[0].unsqueeze(dim=0)
        p = p[0].unsqueeze(dim=0)
        n = n[0].unsqueeze(dim=0)
        print(a,p,n)
        anchor_out,positive_out,negative_out = network(a,p,n)
        break
    
    anchor_out = anchor_out.detach().cpu()
    positive_out = positive_out.detach().cpu()
    negative_out = negative_out.detach().cpu()
    print(anchor_out,positive_out,negative_out)
    print(network.loss_fn(anchor_out,positive_out,negative_out))
    plt.scatter([anchor_out[0,0]],[anchor_out[0,1]],label = 'anchor')
    plt.scatter([positive_out[0,0]],[positive_out[0,1]],label = 'positive')
    plt.scatter([negative_out[0,0]],[negative_out[0,1]],label = 'negative')
    plt.legend()

    plt.savefig(f'what_test.png')

def debug_2():
    all_seen_positives = []
    for batch in train_data:
        p = batch[1]
        count = 0
        for pos in all_seen_positives:
            if torch.equal(p,pos):
                count += 1
        if count > 0:
            print(count)
        all_seen_positives.append(p)
    raise Exception

def test_on_data():
    batch_size = 128
    num_choices = 1
    num_picks = 10
    max_samples = 10000
    max_samples_test = None
    network = models.Siamese_Network(test_choices = num_picks, embedding_dimensions = 256, use_weighting = False)
    network.load_state_dict(torch.load('rbc_network_second_version.pt'))
    network.eval()
    #pick_train_data = models.Siamese_RBC_dataset('debugging/',num_choices = num_picks,max_samples = max_samples)
    pick_train_data = models.Siamese_RBC_dataset('datasets/train/',num_choices = num_picks,max_samples = max_samples)
    network.test_picks = True    
    pick_train_loader = DataLoader(pick_train_data,batch_size,shuffle = False, collate_fn= pick_collate_fn, num_workers=32)
    early_stop_callback = EarlyStopping(monitor="epoch_train_loss", min_delta=0.00, patience=1, verbose=True, mode="min")
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu', callbacks = [early_stop_callback],max_epochs = 1)
    trainer.test(network,dataloaders = pick_train_loader)


def look_at_some_data():
    network = models.Siamese_Network(test_choices = 10, embedding_dimensions = 256, use_weighting = False)
    network.load_state_dict(torch.load('rbc_network_small.pt'))
    network.eval()
    pick_train_data = models.Siamese_RBC_dataset('datasets/backup/',num_choices = 10,max_samples = 10000)
    #print(sum([pick_train_data[i][0][-1,-1,0,0] for i in range(1000,1200)]))
    pick_train_loader = DataLoader(pick_train_data,1,shuffle = False, collate_fn= pick_collate_fn, num_workers=32)
    i = 0
    for anchor,choice_list,lens in pick_train_loader:
        anchors = network.anchor_forward(anchor)
        choices = network.choice_forward(choice_list.view(1*(network.test_choices+1),12,8,8))
        choices = choices.view(1,network.test_choices+1,-1)
        
        distances = torch.stack([models.get_distance(anchors,choices[:,j,:]).cpu() for j in range(network.test_choices+1)])
        with open(f'distances_{i}.csv', 'w', newline='') as f :
            writer = csv.writer(f,delimiter = ';')
            writer.writerow(distances.tolist())
        with open(f'anchor_{i}.csv', 'w', newline='') as f :
            writer = csv.writer(f,delimiter = ';')
            writer.writerows(anchor)
        with open(f'positive_{i}.csv', 'w', newline='') as f :
            writer = csv.writer(f,delimiter = ';')
            writer.writerow(choice_list[0,0,:])
        with open(f'negatives_{i}.csv', 'w', newline='') as f :
            writer = csv.writer(f,delimiter = ';')
            for n in choice_list[0,1:,:]:
                writer.writerow(n)
        i += 1
        if i == 100:
            raise Exception






device_num = 2
device = 'cuda:'+str(device_num)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    batch_size = 1024
    num_workers = 64
    num_choices = 1
    num_picks = 10

    max_samples = None
    max_samples_test = None
    train_path = 'data/siamese_playerlabel/train/'
    val_path = 'data/siamese_playerlabel/val/'
    train_data = models.Siamese_RBC_dataset(train_path,num_choices = num_choices,max_samples = max_samples)
    train_loader = DataLoader(train_data,batch_size,shuffle = True, persistent_workers=True, pin_memory = False, collate_fn= collate_fn_multianchor, num_workers=num_workers)
    val_data = models.Siamese_RBC_dataset(val_path,num_choices = num_choices,max_samples = max_samples)
    val_loader = DataLoader(val_data,batch_size,shuffle = False, persistent_workers=True, pin_memory = False,collate_fn= collate_fn_multianchor, num_workers=num_workers)
    # pick_train_data = models.Siamese_RBC_dataset(train_path,num_choices = num_picks,max_samples = max_samples)
    # pick_val_data = models.Siamese_RBC_dataset(val_path,num_choices = num_picks,max_samples = max_samples)
    # pick_train_loader = DataLoader(pick_train_data,batch_size,shuffle = False, collate_fn= pick_collate_fn_multianchor, num_workers=num_workers)
    # pick_val_loader = DataLoader(pick_val_data,batch_size,shuffle = False, collate_fn= pick_collate_fn_multianchor, num_workers=num_workers)


    print(f'Will train {train_data.num_players} different context players')
    network = models.Siamese_Network_Tweaking(test_choices = num_picks, embedding_dimensions = 512, use_weighting = False)
    early_stop_callback = EarlyStopping(monitor="epoch_eval_loss", min_delta=0.00, patience=2, verbose=True, mode="min")
    trainer = pl.Trainer(devices = [device_num], accelerator = 'gpu', \
        profiler="simple", callbacks = [early_stop_callback],max_epochs = -1)

    
    pr = cProfile.Profile()
    pr.enable()
    trainer.fit(network,train_dataloaders=train_loader, val_dataloaders=val_loader)
    pr.disable()

    torch.save(network.state_dict(),'siamese_players.pt')
