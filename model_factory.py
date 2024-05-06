# -*- coding: utf-8 -*-
'''
@author: xw2941
'''

from lstm_model import LSTMCTCModel
from conformer_model import ConformerModel

def get_model(model_type, config):
    if model_type == 'LSTM':
        return LSTMCTCModel(
            num_features=config['num_features'],
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            num_classes=config['num_classes']
        )
    elif model_type == 'Conformer':
        return ConformerModel(
            num_features=config['num_features'],
            d_model=config['conformer_d_model'],
            num_blocks=config['conformer_num_blocks'],
            n_heads=config['conformer_n_heads'],
            num_classes=config['num_classes']
        )
    else:
        raise ValueError("Unsupported model type")
