# model.py

#CNN = local feature extractor.

#LSTM = temporal dependency learner.

#Attention = focuses on key timesteps.

#FC layer = classifier





import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import random
import numpy as np


# Fix random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class Attention(nn.Module):
    """Simple attention mechanism"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        weights = torch.softmax(self.attn(x), dim=1)  # [batch_size, seq_len, 1] ,single score to each timestamp feature vector
        out = (x * weights).sum(dim=1)  # weighted sum over sequence
        return out

class HybridModel(nn.Module):
    def __init__(self, input_channels, cnn_channels, lstm_hidden, lstm_layers, num_classes):
        super(HybridModel, self).__init__()
        
        
        
        #[batch , input_chennels, seq_len]
        # 1D CNN for temporal patterns 
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=3, padding=1), #feature extractor
            nn.ReLU(), #non_linearity
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)  # Reduce to fixed sequence length
        )
        
        
        
        #[btach , seq_len , input_chennels]
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(input_size=cnn_channels, # features per timestep (from CNN output)
                            hidden_size=lstm_hidden, # number of hidden units in each LSTM cell
                            num_layers=lstm_layers, # how many stacked LSTM layers
                            batch_first=True, # input/output format: [batch, seq_len, features]
                            bidirectional=True) # captures both past and future context.
        
        # Attention
        self.attention = Attention(lstm_hidden*2) #as lstm is bidictional
        
        # Classifier
        self.fc = nn.Linear(lstm_hidden*2, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_channels]
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, seq_len] for Conv1d
        x = self.cnn(x)         # [batch_size, cnn_channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, cnn_channels] for LSTM
        
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden*2]
        attn_out = self.attention(lstm_out)  # [batch_size, hidden*2]
        
        out = self.fc(attn_out)  # [batch_size, num_classes]
        return out