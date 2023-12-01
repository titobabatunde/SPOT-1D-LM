# %%
import time
import torch
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

# %%
# import Datasets and Networks
from dataset.dataset_inference_class import Proteins_Dataset_Class
from dataset.dataset_inference_test import Proteins_Dataset_Test


from models.bilstm import Network
from models.ms_resnet import Network as Network2
from models.ms_res_lstm import Network as Network3
SS3_CLASSES = ['C', 'E', 'H']  # Define your SS3 classes
"""
latest file fixed validate and train method
"""

# %%
# hyperparameters
config = dict(
    file_list_train = "spot_1d_lm/lists/train.txt",
    file_list_val   = "spot_1d_lm/lists/val.txt",
    file_list_test  = "spot_1d_lm/lists/casp12.txt",
    batch_size      = 10,
    epoch           = 150,
    loss            = torch.nn.CrossEntropyLoss(),
    device          = "cuda:3",
    learning_rate   = 1e-3,
    run             = 1
)

# %%
train_dataset       = Proteins_Dataset_Class(
    file_name_list  = config["file_list_train"]
)
valid_dataset       = Proteins_Dataset_Class(
    file_name_list  = config["file_list_val"]
)
test_dataset        = Proteins_Dataset_Test(
    file_name_list  = config["file_list_test"]
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

model1 = Network(input_size=2324)
model2 = Network2(input_channel=2324)
model3 = Network3(input_channel=2324)

# %%
class EnsembleNetwork(torch.nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleNetwork, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.softmax = torch.nn.Softmax(dim=1)
    # end def

    def forward(self, x, x_lens):
        # Get outputs from each model
        out1 = self.model1(x, x_lens)
        out2 = self.model2(x, x_lens)
        out3 = self.model3(x, x_lens)

        # Average the outputs
        avg_out = (out1 + out2 + out3) / 3

        # Apply softmax
        result = self.softmax(avg_out)
        return result
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
model = model.to(config['device'])
print(model)

# %%
optimizer   = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
criterion   = config['loss']
scaler      = torch.cuda.amp.GradScaler()  # Initialize the gradient scaler for mixed-precision training
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.8, 
                                                         patience=3, 
                                                         verbose=True)

# %%
def train(model, dataloader, criterion, optimizer):
    model.train()  # Set the model to training mode

    # Progress Bar
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    total_loss  = 0
    num_correct = 0
    total_samples = 0
    total_correct = 0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()  # zero gradients

        x, y, lengths, protein_names, sequences = batch
        x, y = x.to(config['device']), y.to(config['device'])
        # y shape is [batch_size, sequence_length]

        # -1 is used to represent missing data
        # it is essential that these values are not included in loss calculation
        # mask out these values when calculating the loss


        # Mixed-precision training context
        with torch.cuda.amp.autocast():
            outputs = model(x, lengths)
            # output shapes is [batch_size, sequence_length, num_classes]
            # this needs to be [batch_size, num_classes, sequence_length]
            outputs = outputs.permute(0,2,1)
            # print(f'before masking: outputs shape: {outputs.shape} and y shape: {y.shape}')
            # Create a 3D mask that matches the outputs tensor shape
            valid_data_mask = (y != -1).unsqueeze(1).expand_as(outputs)
            outputs = outputs[valid_data_mask].view(-1, 3)  # Flatten and keep 3 class outputs
            # apply mask to y
            y = y[y != -1]

            loss = criterion(outputs, y)
        # end with

        # print(f'after masking: outputs shape: {outputs.shape} and y shape: {y.shape}')
        # before masking: outputs shape: torch.Size([10, 3, 256]) and y shape: torch.Size([10, 256])
        # after masking: outputs shape: torch.Size([1123, 3]) and y shape: torch.Size([1123])
        batch_acc        = (torch.argmax(outputs, axis=1) == y).sum().item() / y.shape[0]
        total_correct   += (torch.argmax(outputs, axis=1) == y).sum().item()
        total_samples   += y.shape[0]
        total_loss      += float(loss.item())


        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(float(100 * batch_acc)),
            loss        = "{:.04f}".format(float(total_loss / (i + 1))),
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # Update model parameters
        scaler.update()  # Update the scale for next iteration

        # scheduler.step()

        batch_bar.update() # update tqdm
        del x, y, lengths, protein_names, sequences
        torch.cuda.empty_cache()
    
    batch_bar.close()

    acc        = float(100 * total_correct / total_samples)
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss
# end def

# %%
def validate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0
    total_loss  = 0
    total_samples = 0
    total_correct = 0

    # with torch.no_grad():  # Disable gradient computation
    for i, batch in enumerate(dataloader):
        x, y, lengths, protein_names, sequences = batch
        x, y = x.to(config['device']), y.to(config['device'])

        # get model outputs
        with torch.inference_mode():
            outputs = model(x, lengths)
            # Change shape to [batch_size, num_classes, sequence_length]
            outputs = outputs.permute(0, 2, 1)  

            # Create a 3D mask that matches the outputs tensor shape
            valid_data_mask = (y != -1).unsqueeze(1).expand_as(outputs)
            outputs = outputs[valid_data_mask].view(-1, 3)  # Flatten and keep 3 class outputs
            # apply mask to y
            y = y[y != -1]
            
            loss    = criterion(outputs, y)
        # end with

        batch_acc        = (torch.argmax(outputs, axis=1) == y).sum().item() / y.shape[0]
        total_correct   += (torch.argmax(outputs, axis=1) == y).sum().item()
        total_samples   += y.shape[0]
        total_loss      += float(loss.item())

        batch_bar.set_postfix(
            acc         = "{:.04f}%".format(float(100 * batch_acc)),
            loss        = "{:.04f}".format(float(total_loss / (i + 1)))
        )

        batch_bar.update()
    # end for

    batch_bar.close()

    acc        = float(100 * total_correct / total_samples)
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss
# end def

# %%
wandb.login(key="3e9397f29d471b6beecce85c11b0ffc7a75c8296") #API Key is in your wandb account, under settings (wandb.ai/settings)

run = wandb.init(
    name = "project-submission", ## Wandb creates random run names if you skip this field
    reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # run_id = ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "project-ablations", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)

# %%
""" Experiments """
patience            = 15
best_valacc         = 0.0
improvement_count   = 0
delta               = 0.001
epoch_model_path = 'checkpoint.pth' # set the model path( Optional, you can just store best one. Make sure to make the changes below )
best_model_path = 'best_model.pth'# set best model path

for epoch in range(config['epoch']):
    print("\nEpoch: {}/{}".format(epoch+1, config['epoch']))

    # curr_lr = scheduler.float(optimizer.param_groups[0]['lr'])
    curr_lr = optimizer.param_groups[0]['lr']
    train_acc, train_loss = train(model, train_loader, criterion, optimizer)

    print("\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
        epoch + 1,
        config['epoch'],
        train_acc,
        train_loss,
        curr_lr))

    val_acc, val_loss = validate(model, valid_loader)
    scheduler.step(val_acc)

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Loss {:.04f}\t Val Acc {:.04f}%".format(val_loss, val_acc))    

    wandb.log({
        'train_loss': train_loss,
        'valid_loss': val_loss,
        'valid_acc' : val_acc,
        'lr'        : curr_lr
    })  


    if val_acc >= (best_valacc+delta):
        save_model(model, optimizer, scheduler, ['valid_acc', val_acc], epoch, epoch_model_path)
        wandb.save(epoch_model_path)
        print("Saved epoch model")
        best_valacc = val_acc
        improvement_count = 0
    else:
        improvement_count+=1

    if improvement_count >=patience:
        print(f"Early stopping after {epoch+1} epochs due to no improvement in validation accuracy.")
        break 
      # You may find it interesting to exlplore Wandb Artifcats to version your models
run.finish()

# %%
print(config['epochs'])


