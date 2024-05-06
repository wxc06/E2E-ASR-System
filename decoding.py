# -*- coding: utf-8 -*-

'''
@author: xw2941
'''
import jiwer
class GreedyDecoder:
    def __init__(self, labels, blank_label=0):
        self.labels = labels
        self.blank_label = blank_label
        self.space_label = labels.index(' ') if ' ' in labels else -1

    def decode(self, indices):
        decoded_sequences = []
        # print("Indices shape:", indices.shape)
        # print("Indices dtype:", indices.dtype)
        # print("First batch indices sample:", indices[0][:10])  # print index

        for i, prob in enumerate(indices.t()):  
            sequence = []
            last_char = None
            for idx in prob:
                # print("Current index:", idx.item())  # print current index
                if idx != self.blank_label and idx != last_char:
                    char = self.labels[idx.item()]  # convert Index to Char
                    # print("Corresponding char:", char)
                    sequence.append(char)  
                last_char = idx
            # print(sequence)
            decoded_sequences.append(''.join(sequence).replace('<eos>', ''))
            # if i < 3:  # print decode sequence
            #     print(f"Decoded sequence {i+1}: {decoded_sequences[-1]}")
        return decoded_sequences

def calculate_wer(predictions, references):
    """
    Calculate word error rate between predictions and references.
    """
     # Check if predictions or references are empty
    if not predictions or not references:
        return 1.0 if references else 0.0  # Return 1.0 (100% error) if there should be output, 0.0 if both are empty

     # calculate WER
    wer_score = jiwer.wer(predictions, references)
    return wer_score
