# -*- coding: utf-8 -*-

'''
@author: xw2941
'''

""" Dataloader """
batch_size = 24

char_map = {'C': 30, 'M': 3, 'S': 11, 'E': 28, 'K': 10, 'J': 16, "'": 9, 'F': 7, 'T': 27, 'D': 8, 'Q': 29, 'Y': 6, 'V': 13, 'W': 25, ' ': 18, '<eos>': 2, 'R': 12, 'L': 21, 'P': 23, '<sos>': 1, 'U': 22, 'X': 4, '<pad>': 0, 'B': 15, 'H': 14, 'I': 17, 'O': 19, 'N': 24, 'A': 5, 'Z': 20, 'G': 26}

""" device """
device = 'cuda'


model_config = {
    'model_type': 'LSTM',  # or 'LSTM'
    'num_features': 80, # = n_mels
    'num_classes': 31, # = len(char_map)
    'lstm_hidden_size': 256,
    'lstm_num_layers': 3,
    'conformer_d_model': 512,
    'conformer_num_blocks': 16,
    'conformer_n_heads': 8,
    'num_classes': 31

}

""" train """
save_path = "/content/drive/MyDrive/Colab_Notebooks/Speech/Model_2"
num_epochs = 50
learning_rate = 0.001

#
model_path = '/content/drive/MyDrive/Colab_Notebooks/Speech/Model_2/model_epoch_16.pth'
