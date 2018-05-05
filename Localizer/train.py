import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Model.dataloader import NutritionDataset
from Model.net import LocalizerNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype=torch.float32

def train_localizer(model, optimizer, data_loader, epochs=1):
    model = model.to(device)
    for e in range(epochs):
        for train_batch, labels_batch in data_loader:
            x = train_batch.to(device=device, dtype=dtype)
            y = labels_batch.to(device=device, dtype=dtype)
            scores = model(x)
            print(scores[0])

            # loss = calculate_loss(x, y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

def calculate_loss(x, y):
    '''
        x - NCHW
        y - 550 {5x5x2x11}
    '''
    raise NotImplementedError
