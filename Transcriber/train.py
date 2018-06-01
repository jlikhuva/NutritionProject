import os
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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype=torch.float32

LARGE_NUMBER = 1e5
LOG_OF_CAPTIONS = '../Data/FullData/LOG_OF_CAPTIONS.txt'
def pre_train_encoder(
    encoder, optimizer, train_data_loader, dev_data_loader,
    restore_path='best_encoder_model.tar', save=True,
    restore=False, scheduler=None, epochs=1
):
    train_losses, dev_losses, train_acc, dev_accs = [], [], [], []
    best_loss = LARGE_NUMBER
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
    if restore:
        restore_loc = os.path.join('../Data/FullData/', restore_path)
        checkpoint = torch.load(
            restore_loc,
            map_location=lambda storage, loc: storage
        )
        encoder.load_state_dict(checkpoint['encoder'])
        best_loss = checkpoint['loss']
    encoder = encoder.to(device)
    for i in range(epochs):
        for images, _, _, labels in tqdm(train_data_loader):
            images = images.to(device, dtype=dtype)
            _, _, out = encoder(images)
            loss = calculate_encoder_loss(out, labels)
            encoder.zero_grad(); loss.backward();
            optimizer.step()

            with torch.no_grad():
                train_losses.append(loss.item())
                train_acc.append(calculate_accuracy(out, labels))
                dev_loss, dev_acc = evaluate_encoder(encoder, dev_data_loader)
                dev_losses.append(dev_loss); dev_accs.append(dev_acc)
                if (i+1)%5 == 0:
                    print("==== Performance Check === ")
                    print("\t Train Loss = ", loss.item())
                    print("\t Dev Loss = ", dev_loss)
                    print("\t Train BLEU = ", train_acc)
                    print("\t Dev BLEU = ", dev_acc)

    return (train_losses, dev_losses), (train_acc, dev_accs)

def calculate_encoder_loss(preds, truth):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(preds.squeeze(), truth)

def calculate_accuracy(preds, truth):
    sigmoid = torch.nn.Sigmoid()
    preds = torch.tensor([0.0 if i < 0.5 else 1.0 for i in sigmoid(preds)])
    return ((preds == truth).sum() / len(truth)).item()

def evaluate_encoder(encoder, dev_data_loader):
    losses = []; acc = []
    for images, _, _, labels in dev_data_loader:
        _, _, out = encoder(images)
        losses.append(calculate_encoder_loss(out, labels).item())
        acc.append(float(calculate_accuracy(out, labels)))
    return (
        torch.mean(torch.tensor(losses)).item(),
        torch.mean(torch.tensor(acc)).item()
    )


def train_transcriber(
    encoder, decoder, optimizer, train_data_loader,
    dev_data_loader, train_dataset, dev_dataset, epochs=1, restore=False,
    restore_path='best_transcription_model.tar',
    scheduler=None, save=True
):
    train_losses, dev_losses, train_bleu, dev_bleu = [], [], [], []
    best_loss = LARGE_NUMBER
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
    if restore:
        restore_loc = os.path.join('../Data/FullData/', restore_path)
        checkpoint = torch.load(
            restore_loc,
            map_location=lambda storage, loc: storage
        )
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        best_loss = checkpoint['loss']

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    for i in range(epochs):
        for images, captions, lengths, aux_labels in tqdm(train_data_loader):
            # print(aux_labels)
            outputs, targets, train_encodings, true_captions, class_ = (
                forward(images, captions, lengths, encoder, decoder)
            )
            loss = calculate_loss(outputs, targets, class_, aux_labels)
            encoder.zero_grad(); decoder.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            '''Evaluate as We train'''
            avg_dev_loss, dev_blu = (
                evaluate_on_dev(dev_data_loader, encoder, decoder, train_dataset, dev_dataset)
            )
            train_blu = calculate_bleu_score(decoder, train_encodings, true_captions, train_dataset, dev_dataset)
            del outputs, targets, train_encodings, true_captions

            train_losses.append(loss.item())
            dev_losses.append(avg_dev_loss)
            dev_bleu.append(dev_blu)
            train_bleu.append(train_blu)

            if save and avg_dev_loss < best_loss:
                utils.save_checkpoint({
                    'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(),
                    'optim_dict': optimizer.state_dict(), 'loss': avg_dev_loss
                }, name=restore_path)
                best_loss = avg_dev_loss

            if (i+1)%5 == 0:
                print("==== Performance Check === ")
                print("\t Train Loss = ", loss.item())
                print("\t Dev Loss = ", avg_dev_loss)
                print("\t Train BLEU = ", train_blu)
                print("\t Dev BLEU = ", dev_blu)

        if scheduler:
            scheduler.step()

    return train_losses, dev_losses, train_bleu, dev_bleu

def forward(images, captions, lengths, encoder, decoder):
    images = images.to(device=device, dtype=dtype)
    captions = captions.to(device=device)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
    encoding, _, class_ = encoder(images)
    outputs = decoder(encoding, captions, lengths)
    return outputs, targets, encoding, captions, class_


def evaluate_on_dev(loader, encoder, decoder, train_dataset, dev_dataset):
    encoder.eval(); decoder.eval()
    losses = []
    for images, captions, lengths, aux_labels in loader:
        outputs, targets, dev_encodings, true_captions, class_ = (
            forward(images, captions, lengths, encoder, decoder)
        )
        loss = calculate_loss(outputs, targets, class_, aux_labels)
        losses.append(loss.item())
    dev_blu = calculate_bleu_score(decoder, dev_encodings, true_captions, train_dataset, dev_dataset)
    del outputs, targets, dev_encodings, true_captions
    encoder.train(); decoder.train()
    return np.mean(np.array(losses)), dev_blu

def calculate_bleu_score(decoder, features_batch, true_captions, train_dataset, dev_dataset):
    bleu_scores = []
    if isinstance(decoder, torch.nn.DataParallel):
        sampled_ids = decoder.module.sample(features_batch)
    else:
        sampled_ids = decoder.sample(features_batch)
    sampled_ids = sampled_ids.cpu().numpy()
    true_captions = true_captions.cpu().numpy()
    smoothing_func = SmoothingFunction().method1
    with open(LOG_OF_CAPTIONS, 'w+') as log:
        for predicted, truth in zip(sampled_ids, true_captions):
            true_caption = get_words(truth, train_dataset)
            generated_caption = get_words(predicted, dev_dataset)
            bleu_scores.append(sentence_bleu(
                true_caption, generated_caption,
                smoothing_function=smoothing_func
            ))
            log.write(' '.join(true_caption)); log.write('\n')
            log.write(' '.join(generated_caption)); log.write('\n')
            log.write('\t\t\t==================\n')
    del sampled_ids
    return np.mean(np.array(bleu_scores, dtype=np.float32))

def get_words(indexes, dataset):
    words = []
    for idx in indexes:
        word = dataset.get_word(idx)
        words.append(word)
        if word == '<end>': break
    return words

def calculate_loss(y_hat, y_truth, aux_y_hat, aux_y, lambdah=1.5):
    loss_function = nn.CrossEntropyLoss()
    aux_loss_function = nn.BCEWithLogitsLoss()
    aux_y_hat = aux_y_hat.to(device); aux_y = aux_y.to(device)
    return loss_function(y_hat, y_truth) + lambdah*aux_loss_function(aux_y_hat.squeeze(), aux_y)
