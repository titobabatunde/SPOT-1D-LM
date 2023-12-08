import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import bilistm as TwoLayerBLSTM
import ms_res_lstm as MsResNetLSTM
import ms_resnet as MsResNet

PADDING_VALUE=0
HIDDEN_DIM_LSTM = 1024
NUM_CLASSES = 11
# this assumes batch of 1

class EnsembleNetwork(nn.Module):
    def __init__(self, sequence_length, num_classes=NUM_CLASSES):
        super(EnsembleNetwork, self).__init__()
        self.sequence_length = sequence_length
        
        # Initialize the three models with the correct input sizes
        self.model1 = TwoLayerBLSTM(input_size=20, hidden_dim=HIDDEN_DIM_LSTM)
        self.model2 = MsResNet(input_channel=20, layers=[5, 5, 5, 1], num_classes=num_classes)
        self.model3 = MsResNetLSTM(input_channel=20, layers=[5, 5, 5, 1], num_classes=num_classes)

    def forward(self, x, seq_lens):
        # Assume x is of shape [batch_size, sequence_length, 20]
        
        # Pass the input through each model
        output1 = self.model1(x, seq_lens)
        output2 = self.model2(x)
        output3 = self.model3(x, seq_lens)
        
        # Average the logits from each model
        logits = (output1 + output2 + output3) / 3.0
        
        # Apply softmax or another final activation if needed
        return F.log_softmax(logits, dim=-1) #TODO: I don't think they apply a softmax

# x would be of shape [1, L, 20] and seq_lens would be a list [L]
# L = 200  # Example sequence length

# batch_size = 5  # For example, a batch size of 5
# but here batch_size is actually L
# L = 200  # Example sequence length
# x = torch.randn(batch_size, L, 20)
# seq_lens = [L] * batch_size  # All sequences in the batch have length
# ensemble_model = EnsembleNetwork(sequence_length=L)
# logits = ensemble_model(x, seq_lens)
