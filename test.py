# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from decoding import GreedyDecoder, calculate_wer
from model_factory import get_model
import config
from config import model_config

class Tester:
    def __init__(self, model_config, model_path, test_loader, char_map):
        self.model = get_model(model_config['model_type'], model_config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(config.device)
        self.test_loader = test_loader
        self.char_map = char_map

    def test(self):
        self.model.eval()
        total_test_loss = 0
        '''Define labels'''
        num_classes = max(self.char_map.values()) + 1
        labels = [''] * num_classes
        for char, index in self.char_map.items():
            labels[index] = char
        decoder = GreedyDecoder(labels=labels, blank_label=self.char_map['<pad>'])
        criterion = nn.CTCLoss(blank=self.char_map['<pad>'], zero_infinity=True)
        all_predictions, all_references = [], []

        with torch.no_grad():
            for batch_idx, (spectrograms, transcripts, utterances, input_lengths, label_lengths) in enumerate(self.test_loader):
                spectrograms = spectrograms.to(config.device)
                output = self.model(spectrograms)
                output = output.log_softmax(2).permute(1, 0, 2)
                input_lengths = input_lengths.to(config.device)
                loss = criterion(output, transcripts, input_lengths, label_lengths)
                total_test_loss += loss.item()
                decoded_output = decoder.decode(torch.argmax(output, dim=2))
                all_predictions.extend(decoded_output)
                all_references.extend(utterances)
        avg_test_loss = total_test_loss / len(self.test_loader)
        test_wer = calculate_wer(all_predictions, all_references)
        print("Predictions:", all_predictions)
        print("References:", all_references)
        print("Test Loss:", avg_test_loss)
        print("Test WER:", test_wer)

