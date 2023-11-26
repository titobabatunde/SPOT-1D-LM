import time
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os
import gc
import time
from tqdm.notebook import tqdm as blue_tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import wandb
import subprocess

# from config import PATH, LIST, VAL_LIST, TEST_LIST, TEST2_LIST, TEST3_LIST, TEST4_LIST, IGNORE_LABEL, DEVICE

from dataset.dataset_inference_class import Proteins_Dataset_Class
from dataset.dataset_inference_test import Proteins_Dataset_Test


from models.bilstm import Network
from models.ms_resnet import Network as Network2
from models.ms_res_lstm import Network as Network3

# cross Entropy loss here
config = dict(
    file_list_train = "spot_1d_lm/list/train.txt",
    file_list_val   = "spot_1d_lm/list/val.txt",
    file_list_test  = "spot_1d_lm/list/casp12.txt",
    batch_size      = 10,
    epoch           = 100,
    loss            = torch.nn.CrossEntropyLoss(),
    device          = "cuda",
    learning_rate   = 2e-4
)


train_dataset       = Proteins_Dataset_Class(
    file_name_list  = config["file_list_train"]
)
valid_dataset       = Proteins_Dataset_Class(
    file_name_list  = config["file_list_test"]
)
test_dataset        = Proteins_Dataset_Test


test_set = Proteins_Dataset(args.file_list)  ## spot-1d test set
print("test_dataset Loaded with ", len(test_set), "proteins")
# this implementation has only been tested for batch size 1 only.
test_loader = DataLoader(test_set, batch_size=1, collate_fn=text_collate_fn, num_workers=16)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model1 = Network()
model2 = Network2()
model3 = Network3()

# class EnsembleClassNetwork(torch.nn.Module):
#     def __init__(self):
#         super(EnsembleClassNetwork, self).__init__()
#         self.model1 = Network()
#         self.model2 = Network2()
#         self.model3 = Network3()

#     def forward(self, x):
#         # Assuming each model's forward method returns a tensor of the same shape
#         output1 = self.model1(x)
#         output2 = self.model2(x)
#         output3 = self.model3(x)

#         # Average the outputs
#         ensemble_output = (output1 + output2 + output3) / 3
#         return ensemble_output
#     # end def
# # end class

model1.load_state_dict(torch.load("checkpoints/model1.pt", map_location=torch.device('cpu')))
model2.load_state_dict(torch.load("checkpoints/model2.pt", map_location=torch.device('cpu')))
model3.load_state_dict(torch.load("checkpoints/model3.pt",map_location=torch.device('cpu')))


model1 = model1.to(args.device)
model2 = model2.to(args.device)
model3 = model3.to(args.device)



class_out = main_class(test_loader, model1, model2, model3, args.device)
names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out
reg_out = main_reg(test_loader, model4, model5, model6, args.device)
psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out
print(len(ss3_pred_list), len(psi_list))
write_csv(class_out, reg_out, args.save_path)

