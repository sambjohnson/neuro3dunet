import os
import h5py  #

import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def split_volume(np_arr, chunk_size, stride_size):
    """ Splits an np array of shape (nchannels, nx, ny, nz)
    """
    shape = np_arr.shape
    
    assert len(shape) == 4
    assert shape[1] == shape[2] == shape[3]
    
    num_chunks = ((shape[1] - chunk_size) // stride_size) + 1

    chunks_lvl0 = []
    
    # triple loop over x, y, z dimensions of volume 
    for xc in range(num_chunks):
        x0 = xc * stride_size
        x1 = x0 + chunk_size
        
        chunks_lvl1 = []
        for yc in range(num_chunks):
            y0 = yc * stride_size
            y1 = y0 + chunk_size
            
            chunks_lvl2 = []
            for zc in range(num_chunks):
                z0 = zc * stride_size
                z1 = z0 + chunk_size
        
                chunk = np_arr[:, x0:x1, y0:y1, z0:z1]
                chunks_lvl2.append(chunk)
            
            # zoom out 
            chunks_lvl1.append(chunks_lvl2)
        chunks_lvl0.append(chunks_lvl1)

    return chunks_lvl0

def split_datapoint_and_save(h5_dataset, 
                             chunk_size, 
                             stride_size, 
                             save_directory,
                             save_filename_prefix):
    
    """ Splits an h5 file with compatible-shaped
        datasets 'raw' and 'label';
        Returns None.
        
        Its effect is to save out a list of datasets by chunking 'raw' and 'label'
        into a smaller new h5 files.
        
        The newly created h5 files are saved with a filename suffix indicating which
        x, y, z chunk of the original data they contain.
    """    
    
    raw_np = np.array(h5_dataset.get('raw'))
    label_np = np.array(h5_dataset.get('label'))
    
    split_raw_volumes = split_volume(raw_np, chunk_size, stride_size)
    split_label_volumes = split_volume(label_np, chunk_size, stride_size)
    
    nchunks_1d = len(split_raw_volumes[0])  # assumes x, y, z dimensions are equal
    for x in range(nchunks_1d):
        for y in range(nchunks_1d):
            for z in range(nchunks_1d):
                # save out a new h5 file for each chunk of the original volume
                print('chunk:', x, y, z)
                v1 = split_raw_volumes[x][y][z]
                v2 = split_label_volumes[x][y][z]
                
                file = h5py.File(f'{save_directory}/{save_filename_prefix}_chunk_{x}_{y}_{z}.h5', 'w')
                file.create_dataset('raw', data=v1)
                file.create_dataset('label', data=v2)
                file.close()
    
    return

def chunk_entire_dataset(data_in_dir, 
                         data_out_dir, 
                         chunk_size, 
                         stride_size
                        ):
    """
        Takes in a dataset with one h5 volume's worth of data ('raw' and 'label')
        for each subject in 'data_in_dir'
        Populates a directory 'data_out_dir' with a several chunk volumes that are
        obtained from the original volumes. E.g., 8 octants, each their own h5 file,
        for each 1 original h5 file.
    """
    
    h5_filenames = sorted(os.listdir(data_in_dir))
    for h5_filename in h5_filenames:

        chunk_size = 84  # a little under 256 // 3
        stride_size = 56  # with a chunk_size of 84, a stride size of 64 allows for some overlap of chunks.

        h5_in_file = h5py.File(f'{data_in_dir}/{h5_filename}', 'r')
        h5_prefix = h5_filename.split('.')[0] # filter off filename extension .h5

        split_datapoint_and_save(h5_in_file, 
                                 chunk_size, 
                                 stride_size=stride_size,
                                 save_directory=data_out_dir,
                                 save_filename_prefix=h5_prefix
                                )
        
    return


def main():
    chunk_size = 84
    stride_size = 56

    base_dir = "/home/weiner/HCP/projects/CNL_scalpel/h5"
    train_dir = "/home/weiner/HCP/projects/CNL_scalpel/h5/train"
    data_in_dir = train_dir
    data_out_dir = f'{base_dir}/train_chunks/'
    os.makedirs(data_out_dir, exist_ok=True)


    chunk_entire_dataset(data_in_dir=data_in_dir,
                        data_out_dir=data_out_dir, 
                        chunk_size=chunk_size, 
                        stride_size=stride_size)

if __name__ == "__main__":
    main()