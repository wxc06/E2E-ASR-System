# -*- coding: utf-8 -*-

'''
@author: xw2941
'''

import torch
import torch.nn as nn

class LSTMCTCModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, num_classes, bidirectional=True):
        super(LSTMCTCModel, self).__init__()
        # Bi-LSTM Layer
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        # Fully-Connected Layer (hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        # output of LSTM
        outputs, _ = self.lstm(x)
        # output of FC
        outputs = self.fc(outputs)
	# [batch_size, seq_length, num_features]
        return outputs

