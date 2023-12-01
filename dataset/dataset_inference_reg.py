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

class Proteins_Dataset_Reg(Dataset):
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

        # reads the protein sequence from prot_path
        seq = read_fasta_file(prot_path)
        # applies one-hot encdoing to the sequence
        one_hot_enc = one_hot(seq)
        # loads EEM and ProtTrans embeddings from the numpy files
        embedding1 = np.load(os.path.join("inputs/", protein + "_esm.npy"))
        embedding2 = np.load(os.path.join("inputs/", protein + "_pt.npy"))
        # embedding1 = np.load(os.path.join("inputs/", protein + "_pb.npy"))

        # load label data for the protein
        labels = np.load(os.path.join("spot_1d_lm/labels", protein + ".npy"))

        # normalize specific labels
        norm_labels = np.empty((labels.shape[0], 11))
        # normalize specific properties
        norm_labels[:,0] = normalize_asa(labels[:,5], labels[:,3]) # normalize ASA
        norm_labels[:,1] = normalize_hseu(labels[:,7]) # normalize HSE U
        norm_labels[:,2] = normalize_hseu(labels[:,8]) # normalize HSE D

        # normalize dihedral angles
        phi     = normalize_circular_angles(labels[:,9])
        psi     = normalize_circular_angles(labels[:,10])
        theta   = normalize_circular_angles(labels[:,11])
        tau     = normalize_circular_angles(labels[:,12])

        # add dihedral angles into nomalized labels
        norm_labels[:, 3:5] = phi
        norm_labels[:, 5:7] = psi
        norm_labels[:, 7:9] = theta
        norm_labels[:, 9:]  = tau

        # features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)
        # concatenates the one-hot encoded sequence with the two embeddings
        features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)

        protein_len = len(seq)

        # returns a tuple of features, length of protein sequences, 
        # protein name, and protein sequence
        return torch.FloatTensor(features), torch.FloatTensor(norm_labels), protein_len, protein, seq
    # end def

    def text_collate_fn(self, batch):
        """
        collate function for data read from text file
        per batch
        """

        # sort data by protein length in descending order
        # batch.sort(key=lambda x: x[1], reverse=True)

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
        padded_features = nn.utils.rnn.pad_sequence(batch_features, batch_first=True, padding_value=0)
        padded_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)

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