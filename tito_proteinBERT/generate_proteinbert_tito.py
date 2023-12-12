# %%
import os
import time
import os.path
import argparse
import numpy as np
from tqdm import tqdm

# %%
from lxml import etree
from pyfaidx import Faidx
import pandas as pd
from IPython.display import display

# %%
##  @brief  :   Keras & TF Libraries
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# %%
def read_list(file_name):
    """
    read a text file to get the list of elements
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text


def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname + '.fasta', 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA

# %%
##  @brief  :   Local Modules
from proteinbert.tokenization import ADDED_TOKENS_PER_SEQ, index_to_token, token_to_index
from proteinbert.model_generation import ModelGenerator, PretrainingModelGenerator, FinetuningModelGenerator, InputEncoder, load_pretrained_model_from_dump, tokenize_seqs
from proteinbert.existing_model_loading import load_pretrained_model
from proteinbert.finetuning import OutputType, OutputSpec, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

# %%
# file_list = "/mnt/nvme/home/bbabatun/IDL/PROJECT/SPOT-1D-LM/spot_1d_lm/lists/casp12.txt"
# device = "cuda:1"
parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
args = parser.parse_args()
## Need to define a maximum sequence input for model
## Check sequence lengths whilst generating esm embeddings 1100 should be large enough
MAX_SEQ_LEN = 1100  
## Size of Embedding Dim
EMBEDDING_DIM = 1562                     ## > NP Embeddings are size = (max_seq_len, embedding_dim)


# %%
##  @brief  :   Load Model and Tokenizer
pretrained_model_generator, input_encoder = load_pretrained_model()
#input_encoder.to(args.device)
## Lodel model to obtain local_representations & global represntations
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(MAX_SEQ_LEN))
#model.to(args.device) # I

prot_list = read_list(args.file_list)

# %%
# pretrained_model_generator, input_encoder = load_pretrained_model('/mnt/nvme/home/bbabatun/proteinbert_models/epoch_92400_sample_23500000.pkl')
# print(prot_list)

# %%
##  @brief  :   Iterate through Files in Dataset & Generate Embeddings
for prot_path in tqdm(prot_list):

    prot_name = prot_path.split('/')[-1].split('.')[0]
    save_path = "/mnt/nvme/home/bbabatun/IDL/PROJECT/SPOT-1D-LM/inputs/" + prot_name + "_pb.npy"
    # print(save_path)

    ## Check no embedding exists
    # if not os.path.isfile(save_path):
    #     try:  
    ## Extract Protein Sequence as a String & Process through Model
    path = "/mnt/nvme/home/bbabatun/IDL/PROJECT/SPOT-1D-LM/spot_1d_lm/labels"
    labels = np.load(os.path.join(path, prot_name + ".npy"), allow_pickle=True)
    # seq = read_fasta_file(prot_path)
    seq = ''.join(labels[:, 3])

    ## Get raw sequence length
    seq_len = len(seq)

    ## Replace Us with Xs to normalise encoding over models
    seq = seq.replace("U", "X")

    ## Encode Input sequence
    encoded_x = input_encoder.encode_X([seq], MAX_SEQ_LEN)

    ## Obtain local & global embeddings
    local_representations, global_representations = model.predict(encoded_x)
    ##local_representations.to(args.file_list)

    ## Remove padding, end and start tokens
    save_arr = local_representations[0,1:seq_len+1,:]

    ## Save np file
    np.save(save_path, save_arr)
    # print(save_path)
        # except:
        #     #   print("No file available for: ",  prot_name, prot_path)
        #       print("No file available for: ",  prot_name, path)

print(" ProteinBERT embeddings generation completed ... ")


