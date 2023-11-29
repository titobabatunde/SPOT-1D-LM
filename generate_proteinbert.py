import time
start_time = time.time()
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from dataset.data_functions import read_list, read_fasta_file
"""
used to generate embeddings for protein sequences using the ProtTrans model
"""


parser = argparse.ArgumentParser()
# --file_list: Path to a fasta file containing a list of protein sequence files.
# --device: The computing device (e.g., 'cpu' or 'cuda') for running the model.
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
args = parser.parse_args()

# loads tokenizer and model for prot_t5_xl_uniref50
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

# setting up the device for Pytorch
device = torch.device(args.device)

# move the model to the specified device
model = model.to(args.device)
# set the model to evaluation mode (disables dropout layers...)
model = model.eval()

# read the list of protein files from the provided file path
prot_list = read_list(args.file_list)

# iterate through each protein file path
for prot_path in tqdm(prot_list):
    # read the protein sequence from the file
    seq = read_fasta_file(prot_path)
    # extract the protein name from the file path
    prot_name = prot_path.split('/')[-1].split('.')[0]

    # insert spaces between each character in the sequence
    seq_temp = seq.replace('', " ")
    # create a list of sequences
    sequences_Example = [seq_temp]
    # replace certain characters with 'X' for normalization
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    # tokenize the sequence
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

    # conver the tokenized sequences to tensors and move to device
    input_ids = torch.tensor(ids['input_ids']).to(args.device)
    attention_mask = torch.tensor(ids['attention_mask']).to(args.device)
    # generate embeddings with no gradient calculations (for efficiency)
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # move embeddings to cpu if cuda and convert to numpy array
    if args.device == "cpu":
        embedding = embedding.last_hidden_state.numpy()
    else:
        embedding = embedding.last_hidden_state.cpu().numpy()

    # extract and process embeddings for each sequence
    features = []
    for seq_num in range(len(embedding)):
        # calculate the actual sequence length (excluding padding)
        seq_len = (attention_mask[seq_num] == 1).sum()
        # extract the embeddings for the actual sequence length
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    # save embeddings to a numpy file
    np.save("inputs/" + prot_name + "_pt.npy", features[0])
print(" ProtTrans embeddings generation completed ... ")
