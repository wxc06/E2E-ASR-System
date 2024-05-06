# -*- coding: utf-8 -*-

'''
@author: xw2941
'''

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

char_map = {'C': 30, 'M': 3, 'S': 11, 'E': 28, 'K': 10, 'J': 16, "'": 9, 'F': 7, 'T': 27, 'D': 8, 'Q': 29, 'Y': 6, 'V': 13, 'W': 25, ' ': 18, '<eos>': 2, 'R': 12, 'L': 21, 'P': 23, '<sos>': 1, 'U': 22, 'X': 4, '<pad>': 0, 'B': 15, 'H': 14, 'I': 17, 'O': 19, 'N': 24, 'A': 5, 'Z': 20, 'G': 26}

def padify(batch):
    """Pad and batch dataset."""
    spectrograms = [item[0] for item in batch]
    transcripts = [torch.tensor([char_map[char] for char in item[1] if char in char_map]) for item in batch]
    utterances = [item[1] for item in batch]

    padded_spectrograms = pad_sequence(spectrograms, batch_first=True)
    padded_transcripts = pad_sequence(transcripts, batch_first=True)

    input_lengths = torch.tensor([len(spec) for spec in spectrograms])
    label_lengths = torch.tensor([len(trans) for trans in transcripts])

    return padded_spectrograms, padded_transcripts, utterances, input_lengths, label_lengths

def create_data_loaders(train_set, dev_set, test_set, batch_size):
    """ Create data loaders """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padify)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, collate_fn=padify)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padify)
    
    return train_loader, dev_loader, test_loader