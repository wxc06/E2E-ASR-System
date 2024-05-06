# -*- coding: utf-8 -*-
'''
@author: xw2941
'''

import torch
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

class AudioDataProcessor(Dataset):
    """Feature Extraction"""
    def __init__(self, dataset, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, power=2.0):
        self.dataset = dataset
        self.transforms = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=power
            ),
            self.StandardizeSpectrogram()
        )

    class StandardizeSpectrogram(torch.nn.Module):
        """ Standardize Mel Spectrogram """
        def forward(self, spectrogram):
            mean = spectrogram.mean()
            std = spectrogram.std()
            return (spectrogram - mean) / std

    def __getitem__(self, idx):
        waveform, sample_rate, utterance, _, _, _ = self.dataset[idx]
        spectrogram = self.transform_audio(waveform)
        return spectrogram, utterance

    def __len__(self):
        return len(self.dataset)

    def transform_audio(self, waveform):
        """Input is waveform"""
        mel_spectrogram = self.transforms(waveform)
        return mel_spectrogram.squeeze(0).transpose(0, 1)

