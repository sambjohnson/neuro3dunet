import os
import h5py # note: importing h5py multiple times can cause an error
import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# translated from imports of unet3d.model
from unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from unet3d.utils import get_class, number_of_features_per_level
from unet3d.model import UNet3D


import torch
import torch.nn as nn
import torch.optim as optim


class HDF5Dataset(Dataset):

    """ A custom Dataset class to iterate over subjects.
        This Dataset assumes that the data take the following form:
            data_dir/
                -- subject0.hdf5 (file with two datasets)
                    -- x_name: 4D array
                    -- y_name: 4D array
                -- subject1.hdf5 (next file with two datasets)
                    -- ...
        Note also that this directory should not contain any other files
        besides h5 files for subjects intended to be included in this dataset.
        -----
        Arguments:
            data_dir
            x_name
            y_name
            ordered_subject_list
        -----       
        Returns:
            Pytorch index-based Dataset where each sample is an x, y pair of tensors
                corresponding to a 3D T1 scan and a 4D set of anatomical labels (one-hot)
        
    """
    
    def __init__(self, 
                 data_dir, 
                 x_name=None,
                 y_name=None,
                 ordered_subject_list=None):
        
        self.data_dir = data_dir

        # parse default args
        x_name = 'raw' if x_name is None else x_name
        y_name = 'label' if y_name is None else y_name
        self.x_name = x_name
        self.y_name = y_name
        
        # parse subject ordering, if specified
        if ordered_subject_list is None:
            ordered_subject_list = sorted(os.listdir(data_dir))
        self.subjects = ordered_subject_list
        

    def __len__(self):
        return len(self.subjects)
    

    def __getitem__(self, index):
        subject = self.subjects[index]  # Select the current datapoint (subject)    
        h5 = h5py.File(f'{self.data_dir}/{subject}', 'r')
        
        x_np = h5.get(self.x_name)
        y_np = h5.get(self.y_name)
        
        x = t.from_numpy(np.array(x_np))
        y = t.from_numpy(np.array(y_np))
        
        h5.close() # close the h5 file to avoid extra memory usage

        # If necessary, apply any preprocessing or transformations to the data
        # data = ...

        return x, y
    
# specifying "float" may or may not be necessary on the GPU
# but it is required on CPU
def get_device():
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    return device
    


def train(dl_train, 
          dl_val, 
          model, 
          optimizer,
          criterion,
          n_batches=1e3,
          lr_scheduler=None,
          epochs=10
         ):
    """ Function to wrap the main training loop.
    """
    
    device = get_device()  # defined in above section
    
    # parse default parameters
    if lr_scheduler is None:
        default_lr_scheduler_patience = 3
        default_lr_scheduler_factor = 0.1
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            patience=default_lr_scheduler_patience, 
                                                            factor=default_lr_scheduler_factor
                                                           )
  
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in dl_train:
            optimizer.zero_grad()

            # Forward pass
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dl_val:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Compute average loss
        train_loss /= len(dl_train)
        val_loss /= len(dl_val)

        # Update learning rate scheduler
        lr_scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model checkpoint if desired
            

        # Check early stopping condition if desired
        # TODO
    return model
    
def main():

    train_dir = 'data/h5/train'
    val_dir = 'data/h5/val'
    ds_train = HDF5Dataset(data_dir=train_dir)
    ds_val = HDF5Dataset(data_dir=val_dir)

    batch_size = 1

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    print("Loaded Datasets")
    ## Define model
    in_channels = 1
    out_channels = 102
    print(f"Creating model with {in_channels} in channels, and {out_channels} out channels.")
    model = UNet3D(in_channels=in_channels, out_channels=out_channels)

    # specifying "float" may or may not be necessary on the GPU
    # but it is required on CPU

    print("Checking for CUDA availability. . . \n")
    if torch.cuda.is_available():
        # GPU is available
        print("CUDA is available! \nAssigning model to CUDA")
        device = torch.device("cuda")
        model.to(device) 
    else:
        # GPU is not available, fall back to CPU
        device = torch.device("cpu")
        model.to(device, dtype=float) 


    ## Training
    checkpoint_dir = './checkpoints'  # change this based on your OS and preferences
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Defining optimizer for model")
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Other training parameters
    epochs = 10
    lr_scheduler_patience = 3
    lr_scheduler_factor = 0.1
    print(f"\nTraining model\n\nepochs = {epochs}\nlr_scheduler_patience = {lr_scheduler_patience}\nlr_scheduler_factor = {lr_scheduler_factor}")
    train(dl_train, dl_val, model, optimizer, criterion, device)

if __name__ == "__main__":
    main()

