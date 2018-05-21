import os
import torch
import numpy as np
from PIL import Image
from itertools import chain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TranscriptionDataset(Dataset):
    def __init__(
        self, image_dir, annotation_path,
        data_path, word_vec_path, idx_to_term_path,
        term_to_idx_path, split='train', debug=True
    ):
        self.cur_split_images = np.load(data_path).item()[split]
        if debug:
            self.images = [os.path.join(image_dir, '1_' + f) for f in self.cur_split_images[:10]]
            self.images += [os.path.join(image_dir, '0_' + f) for f in self.cur_split_images[:10]]
        else:
            self.images = [os.path.join(image_dir, '1_' + f) for f in self.cur_split_images]
            self.images += [os.path.join(image_dir, '0_' + f) for f in self.cur_split_images]

        self.annotations = np.load(annotation_path).item()
        self.word_vectors = np.load(word_vec_path).item()
        self.index_term_dict = np.load(idx_to_term_path).item()
        self.term_index_dict = np.load(term_to_idx_path).item()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        y = self._get_target_sequence(self.images[idx])
        return image, y

    def _get_target_sequence(self, path):
        name = path[path.rfind('/')+1 : ]
        name = name[:name.rfind('_')]
        annotations = self.annotations[name]

        return [self.index_term_dict[i] for i in annotations]

    def get_dense_vector(self, word_index):
        word = self.term_index_dict[word_index]
        ls = torch.Tensor(
            self.word_vectors[word]
        )
        return ls
