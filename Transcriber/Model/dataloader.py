import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from itertools import chain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append("..")
from Shared import utils

class Vocabulary(object):
    '''
        Wrapper for our vocabulary.
        Inspired by https://github.com/yunjey/pytorch-tutorial
    '''
    def __init__(self, word_vec_path):
        self.word_vect_dict = np.load(word_vec_path).item()
        self.all_words = list(self.word_vect_dict.keys())
        self._add_special_tokens()
        self.word_vectors = [self.word_vect_dict[word] for word in self.all_words]
        self.index_to_word = {i:v for i, v in enumerate(self.all_words)}
        self.word_to_index = {v:i for i, v in enumerate(self.all_words)}

    def __call__(self, word):
        return self.word_to_index[word]

    def __len__(self):
        return len(self.all_words)

    def _add_special_tokens(self):
        tokens = ['<pad>', '<start>', '<end>']
        token_vec = np.zeros(len(self.word_vect_dict['g']))
        for token in tokens:
            self.all_words.append(token)
            self.word_vect_dict[token] = token_vec

    def get_word_vectors(self):
        return nn.Parameter(
            torch.from_numpy(np.array(self.word_vectors, dtype=np.float32))
        )
    def get_word_from_index(self, index):
        return self.index_to_word[index]


class TranscriptionDataset(Dataset):
    def __init__(
        self, image_dir, annotation_path,
        data_path, word_vec_path, mean_path,
        split='train', debug=True
    ):
        self.cur_split_images = np.load(data_path).item()[split]
        if debug:
            self.images = [os.path.join(image_dir, '1_' + f) for f in self.cur_split_images[:5]]
            self.images += [os.path.join(image_dir, '0_' + f) for f in self.cur_split_images[:5]]
        else:
            self.images = [os.path.join(image_dir, '1_' + f) for f in self.cur_split_images]
            self.images += [os.path.join(image_dir, '0_' + f) for f in self.cur_split_images]

        self.annotations = np.load(annotation_path).item()
        self.vocab = Vocabulary(word_vec_path)
        self.mean_path = mean_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        y = self._get_target_sequence(self.images[idx])
        transform = transforms.Compose([
            transforms.ToTensor(),
            utils.SubtructMeanImage(self.mean_path)
        ])
        return transform(image), y

    def _get_target_sequence(self, path):
        name = path[path.rfind('/')+1 : ]
        name = name[:name.rfind('_')]
        annotations = self.annotations[name]

        caption = []
        caption.append(self.vocab('<start>'))
        caption += [self.vocab(i) for i in annotations]
        caption.append(self.vocab('<end>'))
        return torch.Tensor(caption)

    def get_word_vectors(self):
        return self.vocab.get_word_vectors()

    def get_output_size(self):
        return len(self.vocab)
    
    def get_word(self, index):
        return self.vocab.get_word_from_index(index)

def collate_fn(data):
    '''
        Function for creating a batch where each datapoint
        is of a different length.
        Based fully on https://github.com/yunjey/pytorch-tutorial

        Returns:
            images: torch tensor (N, 3, 512, 512)
            targets: torch tensor (N, padded_length)
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, caption in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = caption[:end]
    return images, targets, lengths
