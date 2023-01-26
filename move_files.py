import os
from random import shuffle
import shutil
import tqdm
import multiprocessing as mp
import lzma
import dill
import pickle
from ray.util.multiprocessing import Pool
from argparse import ArgumentParser, BooleanOptionalAction

def move_file(params):
    file, path, outpath = params
    if not '.pt' in file:
        return
    os.rename(f'{path}/{file}',f'{outpath}/{file}')

def move_directory(params):
    path,outpath = params
    print(f'Moving all files from {path} to {outpath}')
    all_files = os.listdir(path)
    with Pool(1024) as p:
        p.map(move_file,[(f,path,outpath) for f in all_files])

def split_data(dir_name, test = False):
    path = f'data/siamese_playerlabel/{dir_name}'
    out_train = f'data/siamese_playerlabel/train/{dir_name}'
    out_val = f'data/siamese_playerlabel/val/{dir_name}'
    out_test = f'data/siamese_playerlabel/test/{dir_name}'
    if os.path.exists(out_train) or os.path.exists(out_test) or os.path.exists(out_val):
        print('Data already split')
        return
    else:
        os.makedirs(out_train)
        os.makedirs(out_val)
        if test:
            os.makedirs(out_test)

    files = os.listdir(path)
    num_files = len(files)
    shuffle(files)
    pool_size = 1024
    

    num_val_files = int(num_files/10)
    if test:
        num_test_files = num_val_files
    else:
        num_test_files = 0
    val_files = files[:num_val_files]
    train_files = files[(num_val_files+num_test_files):]
    test_files = files[num_val_files:(num_val_files+num_test_files)]

    with Pool(pool_size) as p:
        p.map(move_file,[(f,path,out_val) for f in val_files])
    with Pool(pool_size) as p:
        p.map(move_file,[(f,path,out_test) for f in test_files])
    with Pool(pool_size) as p:
        p.map(move_file,[(f,path,out_train) for f in train_files])


def remove_true_board(file):
    with lzma.open(file, 'rb') as f:
        data = pickle.load(f)
    
    anchor = data[0]
    positive = data[1]
    negative = data[2]
    name = data[3]
    try:    
        negative.remove(positive)
    except:
        pass

    with lzma.open(file, 'wb') as f:
        pickle.dump((anchor,positive,negative, name),f)

def check_errors(file):
    try:
        with lzma.open(file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'Cant open {file} with pickle')
        try:
            with lzma.open(file, 'rb') as f:
                data = dill.load(f)
        except Exception as e:
            print(f'Cant open {file} with dill either')
            os.remove(file)
            print(e)

if __name__ == '__main__':
    
    # parser = ArgumentParser()
    # parser.add_argument('-t', '--test', dest = 'test', default=False, action=BooleanOptionalAction)
    # parser = parser.parse_args()
    # split_data('move', parser.test)
    # split_data('sense', parser.test)
    path = 'data/siamese_playerlabel/train/move/'
    files = [path + f for f in os.listdir(path)]
    with Pool(1000) as p:
        p.map(check_errors,files)
    