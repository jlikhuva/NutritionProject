import sys
import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
sys.path.append("..")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from Shared import utils
from Model.dataloader import TranscriptionDataset
from Model import encoder_net, decoder_net

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype=torch.float32

LARGE_NUMBER = 1e5
def train_transcriber(
    encoder, decoder, optimizer, train_data_loader,
    dev_data_loader, epochs=1, restore=True,
    restore_path='../Data/FullData/best_transcription_model.tar',
    scheduler=None, save=True
):
    train_losses, dev_losses, train_bleu, dev_bleu = [], [], [], []
    best_loss = LARGE_NUMBER
    if torch.cuda.device_count() > 1 or restore:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    for i in range(epochs):
        for images, captions, lengths in tqdm(train_data_loader):
            outputs, targets = forward(images, captions, lengths, encoder, decoder)
            loss = calculate_loss(outputs, targets)
            avg_dev_loss = evaluate_on_dev(dev_data_loader, encoder, decoder)
            train_losses.append(loss)
            dev_losses.append(avg_dev_loss)

            encoder.zero_grad(); decoder.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            '''Evaluate as We train'''
            if (i+1)%5 == 0:
                print("==== Performance Check === ")
                print("\t Train Loss = ", loss.item())
                print("\t Dev Loss = ", avg_dev_loss)

        if scheduler:
            scheduler.step()

    return train_losses, dev_losses, train_bleu, dev_bleu

def forward(images, captions, lengths, encoder, decoder):
    images = images.to(device=device, dtype=dtype)
    captions = captions.to(device=device)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    encoding, _ = encoder(images)
    outputs = decoder(encoding, captions, lengths)
    return outputs, targets


def evaluate_on_dev(loader, encoder, decoder):
    encoder.eval(); decoder.eval()
    losses = []
    for images, captions, lengths in loader:
        outputs, targets = forward(images, captions, lengths, encoder, decoder)
        loss = calculate_loss(outputs, targets)
        losses.append(loss.item())
    encoder.train(); decoder.train()
    return np.mean(np.array(losses))

def calculate_loss(y_hat, y_truth):
    loss_function = nn.CrossEntropyLoss()
    return loss_function(y_hat, y_truth)
