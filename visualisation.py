import matplotlib
import utils
from NN_Multiple_Board import NN_MultipleBoard
from models import Supervised_Dataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import models

def visualize_opening():
    agent = NN_MultipleBoard()
    agent.handle_game_start(False,'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','Test')

    agent.handle_opponent_move_result(False, None)
    agent.siamese.last_sense = 33
    agent.handle_sense_result([(24+i,None) for i in range(3)]+[(32+i,None) for i in range(3)]+[(40+i,None) for i in range(3)])

    anchor,boards,boards_pre = agent.siamese.get_embeddings(agent.board_dict.get_boards())
    anchor_list = anchor.repeat(len(boards_pre),1)
        
    distances = models.get_distance(anchor,boards)
    normed_distances = torch.ones_like(distances)/ (distances/torch.min(distances))
    normed_distances /= torch.sum(normed_distances)
    for i,b in enumerate(boards_pre):
        print(f'{b.move_stack[-1]}: {distances[i]}; {normed_distances[i]}')


    labels = ['Obs']+ [b.move_stack[-1] for b in boards_pre]
    input_vectors = torch.cat((anchor.unsqueeze(0),boards),dim=0)
    output_vectors = TSNE(n_components = 2, perplexity= 5).fit_transform(input_vectors)
    
    plt.scatter([o[0] for o in output_vectors],[o[1] for o in output_vectors], color=['red'] + ['blue' for _ in range(len(boards_pre))])
    for i,txt in enumerate(labels):
        plt.annotate(txt, (output_vectors[i][0],output_vectors[i][1]))
    plt.savefig('embedding.png')



def look_at_files(path):
    data = Supervised_Dataset(path)
    keys = [
        'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2',
        'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3',
        'rnbqk1nr/pppp1ppp/4p3/8/1b1PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3',
        'rnbqk1nr/pppp1ppp/4p3/8/3PP3/2b5/PP3PPP/RNBQKBNR w KQkq - 0 4',
        'rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2',
        'rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3',
        'rnbqk1nr/pppp1ppp/8/4p3/1b1PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3',
        'rnbqk1nr/pppp1ppp/8/4p3/3PP3/2b5/PP3PPP/RNBQKBNR w KQkq - 0 4'
            ]
    results = {k:[] for k in keys}
    for fen,capture,turn,outcome in data:
        try:
            results[fen].append(outcome.item())
            #print(fen, outcome)
            if len(results[fen]) > 10000:
                break
        except:
            pass
    for k,v in results.items():
        if len(v) > 0:
            print(f'{k}:{np.mean(v)} with {len(v)} samples')
        else:
            print(f'{k}: No samples')

if __name__ == '__main__':
    look_at_files('data/new/train/')

