import os
import pdb
import h5py # note: importing h5py multiple times can cause an error
import numpy as np
import pandas as pd

import torch as t
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# translated from imports of unet3d.model
from unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_decoders, create_encoders
from unet3d.utils import get_class, number_of_features_per_level
from unet3d.model import UNet3D
from unet3d.losses import BCEDiceLoss

import logging
logging.basicConfig(filename="unet.log", level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S')

import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(checkpoint_dir, epoch, model, optimizer):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    checkpoint_list = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_list:
        return 0  # Start training from epoch 0 if no checkpoints are found

    latest_checkpoint = max(checkpoint_list)
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch']


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
        #x = x[:, :64, :64, :64]
        #y = y[:, :64, :64, :64]

        return x, y, subject

def train(dl_train,
          dl_val,
          model,
          optimizer,
          criterion,
          device,
          with_predictions=False,
          epochs=10,
          lr_scheduler=None,
          output_path='./predictions'
          ):
    """ Note: the default path 'output_path' may be inconvenient. Be sure
        to specify your preferred directory in which to save model predictions.
    """
    if lr_scheduler is None:
        default_lr_scheduler_patience = 3
        default_lr_scheduler_factor = 0.1
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            patience=default_lr_scheduler_patience,
                                                            factor=default_lr_scheduler_factor
                                                            )

    # Training loop
    best_val_loss = float('inf')

    # Load the last saved epoch and optimizer state (if available)
    start_epoch = load_checkpoint(checkpoint_dir, model, optimizer)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        sample_set = 0

        for inputs, labels, subject in dl_train:
            optimizer.zero_grad()
            inputs = inputs.to(torch.bfloat16).to(device, dtype=float)
            labels = labels.to(torch.bfloat16).to(device, dtype=float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Save training predictions
            if with_predictions:
                train_predictions = outputs.cpu().numpy()
                np.save(f'{output_path}/train_predictions_epoch_{epoch}_{subject}.npy', train_predictions)

            inputs.detach()
            labels.detach()
            torch.cuda.empty_cache()
            sample_set += 1
            save_checkpoint(checkpoint_dir, epoch, model, optimizer)
            logging.debug("Saving checkpoint")
        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels, subject in dl_val:  # changed to keep track of index
                inputs = inputs.to(torch.bfloat16).to(device, dtype=float)
                labels = labels.to(torch.bfloat16).to(device, dtype=float)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Save validation predictions
                if with_predictions:
                    val_predictions = outputs.cpu().numpy()
                    np.save(f'{output_path}/val_predictions_epoch_{epoch}_{subject}.npy', val_predictions)

                inputs.detach()
                labels.detach()
                torch.cuda.empty_cache()

        train_loss /= len(dl_train)
        val_loss /= len(dl_val)

        lr_scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return model

def main():
    train_dir = '/home/weiner/bparker/NotBackedUp/train_chunks'
    val_dir = '/home/weiner/bparker/NotBackedUp/test_chunks'

    #train_dir = 'data/h5/train'
    #val_dir = 'data/h5/val'
    # option to use DistributedSampler to distribute data over multiple GPUs

    batch_size = 1

    ds_train = HDF5Dataset(data_dir=train_dir)
    ds_val = HDF5Dataset(data_dir=val_dir)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    print("Loaded Datasets\n")



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

        ### DistributedDataParallel chunk
        #os.environ['MASTER_ADDR'] = 'localhost'
        #os.environ['MASTER_PORT'] = '12345'
        #torch.distributed.init_process_group(backend='nccl',
        #                                     init_method = "env://",
        #                                     world_size = 2,
        #                                     rank = 1)
        #model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1])
        #model = nn.DataParallel(model, device_ids=[0,1])
        model.to(device, dtype=float)
    else:
        # GPU is not available, fall back to CPU
        device = torch.device("cpu")
        model.to(device, dtype=float)



        ### DistributedDataParallel chunk
    ## When this is run, it hangs at this stage
    ## Attempting to distribute the data with DistributedSampler()

    #dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    #dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    #print("Loaded Datasets\n")


    #ds_train = HDF5Dataset(data_dir=train_dir)
    #ds_train = DistributedSampler(ds_train)

    #ds_val = HDF5Dataset(data_dir=val_dir)
    #ds_val = DistributedSampler(ds_val)
    #####

    ## Training

    checkpoint_dir = './checkpoints/BCEDice'  # change this based on your OS and preferences
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Defining optimizer for model")
    optimizer = optim.Adam(model.parameters())
    criterion = BCEDiceLoss(alpha=.67, beta=.33)

    # Other training parameters
    epochs = 10
    lr_scheduler_patience = 3
    lr_scheduler_factor = 0.1
    logging.info("\nTraining model\n\nepochs = {epochs}\nlr_scheduler_patience = {lr_scheduler_patience}\nlr_scheduler_factor = {lr_scheduler_factor}")
    print("\nTraining model\n\nepochs = {epochs}\nlr_scheduler_patience = {lr_scheduler_patience}\nlr_scheduler_factor = {lr_scheduler_factor}")
    #pdb.set_trace()
    train(dl_train, dl_val, model, optimizer, criterion, device, checkpoint_dir)

if __name__ == "__main__":
    main()



