from models import Siamese_Network, Siamese_RBC_dataset, get_distance, board_from_fen
from training_siamese import pick_collate_fn_multianchor, collate_fn_multianchor
from torch.utils.data import DataLoader
from create_data import process_files
import umap
import pytorch_lightning as pl
import numpy as np
import tqdm
from collections import defaultdict
import chess
from sklearn.manifold import TSNE
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import shutil
import cProfile

def pick_accuracy_of_game(game_number):
    path = 'data/tmp_data/'
    if os.path.exists(path+'move/'):
        shutil.rmtree(path+'move/')
    if os.path.exists(path+'sense/'):
        shutil.rmtree(path+'sense/')
    process_files(online = str(game_number), outpath= path)
    pick_val_data = Siamese_RBC_dataset(path,num_choices = None,max_samples = None)
    pick_val_loader = DataLoader(pick_val_data,batch_size = 1, collate_fn= pick_collate_fn_multianchor, num_workers=32)
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu')
    network = Siamese_Network(test_choices = 1000, embedding_dimensions = 256, use_weighting = False, multianchor = True,
        player_distance_weight= 0.05)
    network.test_picks = True
    network.load_state_dict(torch.load('networks/players/siamese_players.pt'))
    network.eval()
    trainer.test(network,dataloaders = pick_val_loader)


def visualize_decision():
    print('Visualizing decision')
    pick_val_data = Siamese_RBC_dataset(val_path,num_choices = None,max_samples = 10)
    pick_val_loader = DataLoader(pick_val_data,batch_size = 1, shuffle = True, collate_fn= pick_collate_fn_multianchor)
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu')


    results = {}
    network = Siamese_Network(test_choices = 5000, embedding_dimensions = 256, use_weighting = False,
        player_distance_weight= 0, path = net_path)
    network.visualize = True
    network.test_picks = True
    print(results)


def board_cloud():
    val_data = Siamese_RBC_dataset(val_path,num_choices = 1,max_samples = 1000)
    val_loader = DataLoader(val_data,batch_size = 1000, shuffle = True, collate_fn= collate_fn_multianchor)
    network = Siamese_Network(test_choices = 1, embedding_dimensions = 256, use_weighting = False,
        player_distance_weight= 0, path = net_path).to(device)
    players = defaultdict(list)
    outputs = []
    j = 0
    for anchors,positives,negatives,negative_lens,encodings in tqdm.tqdm(val_loader):
        with torch.no_grad():
            output = network.choice_forward(positives.to(device))
            for i in range(anchors.shape[0]):
                if torch.sum(encodings[i,:,:,:]).item() == 0:
                    players[50].append(j)
                else:
                    players[torch.argmax(encodings[i,:,:,:],dim=0)[0,0].item()].append(j)
                outputs.append(output[i,:])
                j += 1

    reverse_players = {}
    all_colors = mpl.colormaps['viridis'](np.linspace(0, 1, len(players.keys())))
    player_colors = {}
    i = 0
    for k,v in players.items():
        player_colors[k] = all_colors[i]
        for value in v:
            reverse_players[value] = k
        i += 1
    embeddings_colors = [player_colors[reverse_players[k]] for k in range(len(outputs))]
    outputs = torch.stack(outputs)
    for n in [5,10,20,25,50,100,200,500]:
        fit = umap.UMAP(n_neighbors = n)
        things_transformed = fit.fit_transform(outputs.cpu().numpy())
        plt.scatter(things_transformed[:,0],things_transformed[:,1], c = embeddings_colors, s = 5, alpha = 0.5)
        plt.savefig(f'Siamese_point_cloud_{n}neighbors.png')
    

    






if __name__ == "__main__":
    val_path = 'data/siamese_playerlabel/val/'
    net_path = 'networks/players/siamese_players_v126.pt'
    device = 'cuda:1'
    board_cloud()
    # if args.file:
    #     pick_accuracy_of_game(args.file)
    # elif args.viz:
    #     visualize_decision()
    # else:
    #     pick_val_data = Siamese_RBC_dataset(val_path,num_choices = None,max_samples = 2000)
    #     pick_val_loader = DataLoader(pick_val_data,batch_size = 1, collate_fn= pick_collate_fn_multianchor)
    #     trainer = pl.Trainer(devices = [2], accelerator = 'gpu')



    #     weight = 0
    #     network = Siamese_Network(test_choices = 5000, embedding_dimensions = 256, use_weighting = False,
    #         player_distance_weight= weight)
    #     network.test_picks = True
    #     network.load_state_dict(torch.load(net_path))
    #     trainer.test(network,dataloaders = pick_val_loader)