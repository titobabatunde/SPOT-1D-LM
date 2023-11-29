import time
import os.path
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataset.data_functions import read_list, read_fasta_file

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
args = parser.parse_args()