# %%
import time
import torch
import random
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os
import gc
import time
import sys
from tqdm.notebook import tqdm as blue_tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import wandb
import subprocess
from scipy.stats import pearsonr

# %%
# import Datasets and Networks
from dataset.dataset_inference_class_ss8 import Proteins_Dataset_Class
from dataset.dataset_inference_test import Proteins_Dataset_Test

from models.bilstm_reg import Network
from models.ms_resnet_reg import Network as Network2
from models.ms_res_lstm_reg import Network as Network3

REGRESSION = ['ASA', 'HSE UP', 'HSE DOWN', 'PHI_SIN', 'PHI_COS', 'PSI_SIN', 'PSI_COS', 'THETA_SIN', 'THETA_COS', 'TAU_SIN', 'TAU_COS']
"""
latest file fixed validate and train method
"""
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)

# %%
# hyperparameters
config = dict(
    file_path       = "spot_1d_lm/lists/",
    file_list_data  = "files.txt",
    file_list_train = "train.txt",
    file_list_val   = "val.txt",
    file_list_test  = "casp12.txt",
    batch_size      = 10,
    epoch           = 150,
    loss            = torch.nn.L1Loss().to(DEVICE),
    learning_rate   = 1e-3,
    run             = 5
)

def read_and_split_file(file_path, file_name_lists, train_ratio=0.8):
    # Read the list of protein names
    with open(os.path.join(file_path, file_name_lists), 'r') as file:
        protein_names = file.readlines()
    
    # Remove any trailing newline characters
    protein_names = [name.strip() for name in protein_names]

    # Shuffle the list
    random.shuffle(protein_names)

    # Calculate the split index
    split_index = int(len(protein_names) * train_ratio)

    # Split the list into training and validation
    train_list = protein_names[:split_index]
    val_list = protein_names[split_index:]

    # Save the training and validation lists
    with open(os.path.join(file_path, 'train.txt'), 'w') as file:
        for name in train_list:
            file.write(name + '\n')

    with open(os.path.join(file_path, 'val.txt'), 'w') as file:
        for name in val_list:
            file.write(name + '\n')
# end def

read_and_split_file(config['file_path'], config['file_list_data'])


# %%
train_dataset       = Proteins_Dataset_Class(
    file_name_list  = os.path.join(config['file_path'], config["file_list_train"])
)
valid_dataset       = Proteins_Dataset_Class(
    file_name_list  = os.path.join(config['file_path'], config["file_list_val"])
)
test_dataset        = Proteins_Dataset_Test(
    file_name_list  = os.path.join(config['file_path'], config["file_list_test"])
)

gc.collect()
train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config['batch_size'],
    shuffle     = True,
    num_workers = 4,
    pin_memory  = True,
    collate_fn  = train_dataset.text_collate_fn
)

valid_loader    = DataLoader(
    dataset     = valid_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = valid_dataset.text_collate_fn
)

test_loader     = DataLoader(
    dataset     = test_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True,
    collate_fn  = test_dataset.text_collate_fn
)

# %%
print("No. of train proteins   : ", train_dataset.__len__())
print("Batch size           : ", config['batch_size'])
print("Train batches        : ", train_loader.__len__())
print("Valid batches        : ", valid_loader.__len__())
print("Test batches         : ", test_loader.__len__())

print("\nChecking the shapes of the data...")
for batch in train_loader:
    x, y, lens, protein_name, sequence = batch
    print(x.shape, y.shape, lens.shape) 
    # print('protein names in batch')
    # print(protein_name)
    # print('sequences in batch')
    # print(sequence)
    break

# %%
# TRAINING SETUP
def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict(),
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

def load_model(best_path, epoch_path, model, mode= 'best', metric= 'valid_acc', optimizer= None, scheduler= None):

    if mode == 'best':
        checkpoint  = torch.load(best_path)
        print("Loading best checkpoint: ", checkpoint[metric])
    else:
        checkpoint  = torch.load(epoch_path)
        print("Loading epoch checkpoint: ", checkpoint[metric])

    model.load_state_dict(checkpoint['model_state_dict'], strict= False)

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #optimizer.param_groups[0]['lr'] = 1.5e-3
        optimizer.param_groups[0]['weight_decay'] = 1e-5
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = torch.load(best_path)[metric]

    return [model, optimizer, scheduler, epoch, metric]
# end def

# %%
torch.cuda.empty_cache()
gc.collect()

model1 = Network(input_size=2862, num_classes=len(REGRESSION))
model2 = Network2(input_channel=2862, num_classes=len(REGRESSION))
model3 = Network3(input_channel=2862, num_classes=len(REGRESSION))

# %%
class EnsembleNetwork(torch.nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleNetwork, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
    # end def

    def forward(self, x, x_lens):
        # Get outputs from each model
        out1 = self.model1(x, x_lens)
        out2 = self.model2(x, x_lens)
        out3 = self.model3(x, x_lens)

        # # Apply softmax
        # result = self.softmax(avg_out)
        return out1, out2, out3
    # end def
# end class

# %%
torch.cuda.empty_cache()
gc.collect()

def initialize_weights(tensor):
    if type(tensor) == torch.nn.Conv1d or type(tensor) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(tensor.weight.data)
    # end if
# end def

model = EnsembleNetwork(model1, model2, model3)
model.apply(initialize_weights)
model = model.to(DEVICE)
print(model)

# %%
optimizer   = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion   = config['loss']
scaler      = torch.cuda.amp.GradScaler()  # Initialize the gradient scaler for mixed-precision training
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=5, 
                                                         verbose=True)
# scheduler    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10)

# %%

def calculate_mae_for_column(output, target, criterion):
    # Mask where either output or target is NaN
    valid_mask = ~torch.isnan(output) & ~torch.isnan(target)
    output_valid = output[valid_mask]
    target_valid = target[valid_mask]

    # Calculate loss based on the specified type
    loss = criterion(output_valid, target_valid) if len(output_valid) > 0 else torch.tensor(0.0)

    mae = torch.nn.functional.l1_loss(output_valid, target_valid, reduction='mean') if len(output_valid) > 0 else torch.tensor(0.0)

    return loss, mae
# end def

def calculate_pcc_for_column(output, target, criterion):
    # Mask where either output or target is NaN
    valid_mask = ~torch.isnan(output) & ~torch.isnan(target)
    output_valid = output[valid_mask]
    target_valid = target[valid_mask]

    # Calculate loss based on the specified type
    loss = criterion(output_valid, target_valid) if len(output_valid) > 0 else torch.tensor(0.0)

    pearson_corr = pearsonr(output_valid.cpu(), target_valid.cpu())[0] if len(output_valid) > 0 else 0

    return loss, pearson_corr

def train(model, dataloader, criterion, optimizer):
    model.train()  # Set the model to training mode
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    running_loss = 0.0
    maes = []
    pearsons = []
    losses = []
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        x, y, lengths, protein_names, sequences = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Replace -1 in y with NaN
        y[y == -1] = float('nan')

        with torch.cuda.amp.autocast():
            output1, output2, output3 = model(x, lengths)
            outputs = torch.stack([output1, output2, output3], dim=0)

            batch_maes=[]
            batch_pearsons=[]
            total_loss = torch.tensor(0.0, device=DEVICE)

            # Calculate metrics for columns 0, 1, and 2 (mean loss)
            for col in range(3):
                output_col = torch.mean(outputs[:, :, col], dim=0)  # Mean across models for this column
                # Mean for ASA, HSE UP, HSE DOWN
                y_col = y[:, col]
                loss, pearson_corr = calculate_pcc_for_column(output_col, y_col, criterion, loss_type='mean')
                # Log or accumulate the metrics
                total_loss += loss # loss is summed instead of mean
                batch_pearsons.append(pearson_corr)
            # end for

            # Calculate metrics for columns 3 and up (median loss)
            for col in range(3, y.shape[1]):
                # Median for angles
                output_col = torch.median(outputs[:, :, col], dim=0).values  # Median across models for this column
                y_col = y[:, col]
                loss, mae = calculate_mae_for_column(output_col, y_col, criterion, loss_type='median')
                # Log or accumulate the metrics
                total_loss += loss # loss is summed instead of mean
                batch_maes.append(mae)
            # end for
        # end with

        # Convert batch_pearsons and batch_maes to string for displaying
        batch_pearsons_str = ', '.join(f"{p:.4f}" for p in batch_pearsons)
        batch_maes_str = ', '.join(f"{m:.4f}" for m in batch_maes)

        # Update the progress bar
        batch_bar.set_postfix(
            loss="{:.04f}".format(total_loss.item()),
            pccs=f"[{batch_pearsons_str}]",
            maes=f"[{batch_maes_str}]",
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

        pearsons.append(batch_pearsons)
        maes.append(batch_maes)
        running_loss += total_loss.item()

        del x, y, lengths, protein_names, sequences
        torch.cuda.empty_cache()


        # Backpropagation
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    batch_bar.close()

    return np.array(maes).mean(axis=0), np.array(pearsons).mean(axis=0), running_loss / len(dataloader)
# end def


# %%

def validate(model, dataloader, criterion):
    model.eval()  # Set the model to training mode
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Val', ncols=5)

    running_loss = 0.0
    maes = []
    pearsons = []
    losses = []
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        x, y, lengths, protein_names, sequences = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Replace -1 in y with NaN
        y[y == -1] = float('nan')

        with torch.inference_mode():
            output1, output2, output3 = model(x, lengths)
            outputs = torch.stack([output1, output2, output3], dim=0)

            batch_maes=[]
            batch_pearsons=[]
            total_loss = torch.tensor(0.0, device=DEVICE)

            # Calculate metrics for columns 0, 1, and 2 (mean loss)
            for col in range(3):
                output_col = torch.mean(outputs[:, :, col], dim=0)  # Mean across models for this column
                # Mean for ASA, HSE UP, HSE DOWN
                y_col = y[:, col]
                loss, pearson_corr = calculate_pcc_for_column(output_col, y_col, criterion, loss_type='mean')
                # Log or accumulate the metrics
                total_loss += loss # loss is summed instead of mean
                batch_pearsons.append(pearson_corr)
            # end for

            # Calculate metrics for columns 3 and up (median loss)
            for col in range(3, y.shape[1]):
                # Median for angles
                output_col = torch.median(outputs[:, :, col], dim=0).values  # Median across models for this column
                y_col = y[:, col]
                loss, mae = calculate_mae_for_column(output_col, y_col, criterion, loss_type='median')
                # Log or accumulate the metrics
                total_loss += loss # loss is summed instead of mean
                batch_maes.append(mae)
            # end for
        # end with

        # Convert batch_pearsons and batch_maes to string for displaying
        batch_pearsons_str = ', '.join(f"{p:.4f}" for p in batch_pearsons)
        batch_maes_str = ', '.join(f"{m:.4f}" for m in batch_maes)

        # Update the progress bar
        batch_bar.set_postfix(
            loss="{:.04f}".format(total_loss.item()),
            pccs=f"[{batch_pearsons_str}]",
            maes=f"[{batch_maes_str}]",
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        batch_bar.update()

        pearsons.append(batch_pearsons)
        maes.append(batch_maes)
        running_loss += total_loss.item()

        del x, y, lengths, protein_names, sequences
        torch.cuda.empty_cache()

    batch_bar.close()

    return np.array(maes).mean(axis=0), np.array(pearsons).mean(axis=0), running_loss / len(dataloader)
# end def

# %%
wandb.login(key="3e9397f29d471b6beecce85c11b0ffc7a75c8296") #API Key is in your wandb account, under settings (wandb.ai/settings)

run = wandb.init(
    name = "post-mask-rlroplateau-5", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "project-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

# %%
""" Experiments """
# remove early stopping
patience            = 15
best_valloss        = float("inf")
improvement_count   = 0
delta               = 0.0001
epoch_model_path = 'checkpoint.pth' # set the model path( Optional, you can just store best one. Make sure to make the changes below )
best_model_path = 'best_model.pth'# set best model path

for epoch in range(config['epoch']):
    print("\nEpoch: {}/{}".format(epoch+1, config['epoch']))

    # curr_lr = scheduler.float(optimizer.param_groups[0]['lr'])
    curr_lr = optimizer.param_groups[0]['lr']
    mean_mae, mean_pcc, train_loss = train(model, train_loader, criterion, optimizer)

    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epoch'],
        train_loss,
        curr_lr))

    vmean_mae, vmean_pcc, val_loss = validate(model, valid_loader)
    scheduler.step(val_loss)
    # scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Loss {:.04f}\t Val Acc {:.04f}%".format(val_loss))    

    wandb.log({
        'train_loss': train_loss,
        'valid_loss': val_loss,
        'valid_mae' : vmean_mae,
        'valid_pcc' : vmean_pcc,
        'lr'        : curr_lr
    })  


    if val_loss <= best_valloss:
        save_model(model, optimizer, scheduler, ['valid_loss', val_loss], epoch, f'checkpoint_pb_{config["run"]}_{epoch}.pth')
        wandb.save(epoch_model_path)
        print("Saved epoch model")
        best_valacc = val_loss
    #     improvement_count = 0
    # else:
    #     improvement_count+=1

    # if improvement_count >=patience:
    #     print(f"Early stopping after {epoch+1} epochs due to no improvement in validation accuracy.")
    #     break 
      # You may find it interesting to exlplore Wandb Artifcats to version your models
run.finish()

# %%
print(config['epoch'])


