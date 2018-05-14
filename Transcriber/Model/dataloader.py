import os
import torch
import numpy as np
from PIL import Image
from itertools import chain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TranscriptionDataset(Dataset):
    def __init__(self,):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
