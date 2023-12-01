import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
# imports Dataset class from PyTorch for creating custom data loaders
from dataset.data_functions import one_hot, read_list, read_fasta_file, normalize_asa, normalize_circular_angles, normalize_hsed, normalize_hseu
# custom functions for data processing
# ['AA NAME', 'CHAIN ID', 'RES NUM', 'AA CODE', 'SS3', 'ASA', 'HSE TOTAL', 'HSE UP', 'HSE DOWN', 'PHI', 'PSI', 'THETA', 'TAU', 'OMEGA']

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
SS3_CLASSES = ['C', 'E', 'H']  # Define your SS3 classes

class Proteins_Dataset_Class(Dataset):
    def __init__(self, file_name_list):
        # list is the file path to a list of protein sequences
        # these file path to a list of protein sequences are read
        # protein_list is a list of file paths to fasta per protein
        self.protein_list = read_list(file_name_list)

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        # file path for the protein at index idx
        prot_path = self.protein_list[idx]
        # extracts the protein name from the protein path
        protein = prot_path.split('/')[-1].split('.')[0]

        # load label data for the protein
        labels = np.load(os.path.join("spot_1d_lm/labels", protein + ".npy"), allow_pickle=True)

        # reads the protein sequence from prot_path
        # seq = read_fasta_file(prot_path)
        seq = ''.join(labels[:, 3])
        # print(seq)
        # applies one-hot encdoing to the sequence
        one_hot_enc = one_hot(seq)
        # print(one_hot_enc.shape) # (307, 20)
        # loads EEM and ProtTrans embeddings from the numpy files
        embedding1 = np.load(os.path.join("inputs/", protein + "_esm.npy"))
        embedding2 = np.load(os.path.join("inputs/", protein + "_pt.npy"))
        # embedding1 = np.load(os.path.join("inputs/", protein + "_pb.npy"))



        # normalize specific labels
        ss3_indices = np.array([SS3_CLASSES.index(aa) if aa in SS3_CLASSES else -1 for aa in labels[:, 4]])


        # features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)
        # concatenates the one-hot encoded sequence with the two embeddings
        features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)

        protein_len = len(seq)

        # returns a tuple of features, length of protein sequences, 
        # protein name, and protein sequence
        return torch.FloatTensor(features), torch.LongTensor(ss3_indices), protein_len, protein, seq
    # end def

    def text_collate_fn(self, batch):
        """
        collate function for data read from text file
        per batch
        """

        # sort data by protein length in descending order
        batch.sort(key=lambda x: x[2], reverse=True)

        batch_features, batch_labels, protein_lengths = [], [], []
        protein_names, sequences = [], []

        # unpacks the sorted data into features, protein_len, and sequence
        # features, labels, protein_len, protein, seq = zip(*data)
        for features, labels, protein_len, protein, seq in batch:
            batch_features.append(features)
            batch_labels.append(labels)
            protein_lengths.append(protein_len)
            protein_names.append(protein)
            sequences.append(seq)
        # end for

        # Pad feature and label tensors to ensure they have the same shape
        # enforce_sorted=True
        # Pad label tensors with -1 (or another invalid class index)
        padded_features = nn.utils.rnn.pad_sequence(batch_features, batch_first=True, padding_value=0)
        padded_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-1)

        # returns the padded features, protein lengths,
        # protein names, and sequences
        return padded_features, padded_labels, torch.tensor(protein_lengths), protein_names, sequences
    # end def
# end class


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