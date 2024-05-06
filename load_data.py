# -*- coding: utf-8 -*-

'''
@author: xw2941
'''

import torch
import torchaudio
from torch.utils.data import ConcatDataset

def load_datasets():
    """Load LibriSpeech datasets."""
    train_clean_100 = torchaudio.datasets.LIBRISPEECH(
        root='./',
        url='train-clean-100',
        download=True
    )

    train_clean_360 = torchaudio.datasets.LIBRISPEECH(
        root='./',
        url='train-clean-360',
        download=True
    )

    # Merge train Dataset
    train_dataset = ConcatDataset([train_clean_100, train_clean_360])
    
    dev_dataset = torchaudio.datasets.LIBRISPEECH(
        root='./',
        url='dev-clean',
        download=True
    )
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        root="./",
        url="test-clean",
        download=True
    )
    return train_dataset, dev_dataset, test_dataset

def load_single_dataset(dataset_type):
    assert dataset_type in ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean'], "Invalid dataset type"
    if dataset_type.startswith('train'):
        if dataset_type == 'train-clean-100':
            dataset = torchaudio.datasets.LIBRISPEECH(root='./', url='train-clean-100', download=True)
        else:
            dataset = torchaudio.datasets.LIBRISPEECH(root='./', url='train-clean-360', download=True)
        return dataset
    else:
        dataset = torchaudio.datasets.LIBRISPEECH(root='./', url=dataset_type, download=True)
        return dataset