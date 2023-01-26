import os
from random import shuffle
import shutil
import tqdm
import multiprocessing as mp
from create_data import remove_single_boards



# path = 'datasets/new_new_data/move/'
# for file in tqdm.tqdm(os.listdir(path)):
#     shutil.copyfile(path+file,f'datasets/backup/{file}')




# path = 'datasets/backup/'
# files = os.listdir(path)
# len_before = len(files)
# print(f'Length before: {len_before}')


# with mp.Pool(256) as p:
#     p.map(remove_single_boards,[path+file for file in files])


# len_after = len(os.listdir(path))
# print(f'Length after: {len_after}')


path = 'datasets/backup/'
files = os.listdir(path)
num_files = len(files)
shuffle(files)
for i,file in enumerate(files):
    if not '.pt' in file:
        continue
    if i < num_files/10:
        os.rename(path+file,f'{path}/test/{file}')
    else:
        os.rename(path+file,f'{path}/train/{file}')