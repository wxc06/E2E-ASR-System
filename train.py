# -*- coding: utf-8 -*-

'''
@author: xw2941
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from decoding import GreedyDecoder, calculate_wer

def train_model(model, train_loader, val_loader, char_map, num_epochs, learning_rate, device, save_path):
    criterion = nn.CTCLoss(blank=char_map['<pad>'], zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8)
    train_losses = []
    val_losses = []
    val_wer_scores = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0

        for batch_idx, (spectrograms, transcripts, utterances, input_lengths, label_lengths) in enumerate(train_loader):
            spectrograms, transcripts = spectrograms.to(device), transcripts.to(device)
            optimizer.zero_grad()
            output = model(spectrograms)
            output = output.log_softmax(2).permute(1, 0, 2)  # Correct shape for CTC Loss: [seq_length, batch_size, num_classes]
            # Ensure label_lengths is a tensor and matches batch size
            label_lengths = label_lengths.to(device)

            loss = criterion(output, transcripts, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            if (batch_idx + 1) % 1000 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item()}')

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_wer = validate_model(model, val_loader, criterion, device, char_map)
        val_losses.append(val_loss)
        val_wer_scores.append(val_wer)

        scheduler.step(val_loss)

        torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
        print(f'Epoch: {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}, Validation WER: {val_wer}')
        print(f'Total time for epoch {epoch+1}: {time.time() - start_time} seconds')
    
    # Save the logs after all epochs are completed
    logs = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Validation WER': val_wer_scores
    })
    logs.to_csv(os.path.join(save_path, 'training_logs.csv'), index=False)

def validate_model(model, val_loader, criterion, device, char_map):
    model.eval()
    total_val_loss = 0

    # 创建一个足够大的列表，以最大的索引为准
    num_classes = max(char_map.values()) + 1
    labels = [''] * num_classes  # 使用空字符串初始化
    # 填充列表
    for char, index in char_map.items():
        labels[index] = char
    decoder = GreedyDecoder(labels = labels, blank_label=char_map['<pad>'])

    all_predictions, all_references = [], []
    print("Validation start.")
    with torch.no_grad():
        for batch_idx, (spectrograms, transcripts, utterances, input_lengths, label_lengths) in enumerate(val_loader):
            spectrograms = spectrograms.to(device)
            output = model(spectrograms)
            output = output.log_softmax(2).permute(1, 0, 2)  # [batch_size, seq_length, num_classes]
            input_lengths = input_lengths.to(device)
            loss = criterion(output, transcripts, input_lengths, label_lengths)

            total_val_loss += loss.item()
            decoded_output = decoder.decode(torch.argmax(output, dim=2))
            all_predictions.extend(decoded_output)
            all_references.extend(utterances)
            # print(f"Batch {batch_idx+1} validation loss: {loss.item()}")

    avg_val_loss = total_val_loss / len(val_loader)
    print("Predictions:", all_predictions)
    print("References:", all_references)
    val_wer = calculate_wer(all_predictions, all_references)
    print(f"Validation complete.")
    return avg_val_loss, val_wer
