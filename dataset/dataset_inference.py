import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
# imports Dataset class from PyTorch for creating custom data loaders
from dataset.data_functions import one_hot, read_list, read_fasta_file
# custom functions for data processing

# def one_hot(seq):
#     RNN_seq = seq
#     BASES = 'ARNDCQEGHILKMFPSTWYV'
#     bases = np.array([base for base in BASES])
#     feat = np.concatenate(
#         [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
#          in RNN_seq])
#
#     return feat
#
#
# def read_list(file_name):
#     '''
#     returns list of proteins from file
#     '''
#     with open(file_name, 'r') as f:
#         text = f.read().splitlines()
#     return text
#
#
# def read_fasta_file(fname):
#     """
#     reads the sequence from the fasta file
#     :param fname: filename (string)
#     :return: protein sequence  (string)
#     """
#     with open(fname, 'r') as f:
#         AA = ''.join(f.read().splitlines()[1:])
#     return AA
#

class Proteins_Dataset(Dataset):
    def __init__(self, list):
        # list is the file path to a list of protein sequences
        # these file path to a list of protein sequences are read
        # protein_list is a list of file paths to fasta per protein
        self.protein_list = read_list(list)

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        # file path for the protein at index idx
        prot_path = self.protein_list[idx]
        # extracts the protein name from the protein path
        protein = prot_path.split('/')[-1].split('.')[0]

        # load label data for the protein
        labels = np.load(os.path.join("spot_1d_lm/labels", protein + ".npy"), allow_pickle=True)

        # # reads the protein sequence from prot_path
        # seq = read_fasta_file(prot_path)
        seq = ''.join(labels[:, 3])

        # applies one-hot encdoing to the sequence
        one_hot_enc = one_hot(seq)
        # loads EEM and ProtTrans embeddings from the numpy files
        embedding1 = np.load(os.path.join("inputs/", protein + "_esm.npy"))
        embedding2 = np.load(os.path.join("inputs/", protein + "_pt.npy"))
        # embedding1 = np.load(os.path.join("inputs/", protein + "_pb.npy"))

        # features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)
        # concatenates the one-hot encoded sequence with the two embeddings
        features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)

        protein_len = len(seq)

        # returns a tuple of features, length of protein sequences, 
        # protein name, and protein sequence
        return features, protein_len, protein, seq


def text_collate_fn(data):
    """
    collate function for data read from text file
    """

    # sort data by protein length in descending order
    data.sort(key=lambda x: x[1], reverse=True)
    # unpacks the sorted data into features, protein_len, and sequence
    features, protein_len, protein, seq = zip(*data)
    # converts each feature into a PyTorch float tensor
    features = [torch.FloatTensor(x) for x in features]

    # Pad feature tensots to ensure they have the same shape
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    # returns the padded featyres, protein lengths,
    #  protein names, and sequences
    return padded_features, protein_len, protein, seq


"""
.DSSP FILE
PROTEIN NAME
AA CODE
PHI
PSI
ASA

.T FILE
RES NUM, AA CODE, THETA, TAU, OMEGA

.H FILE
AA NAME, CHAIN ID, RES NUM, AA CODE, HSE TOTAL, HSE UP, HSE DOWN
"""