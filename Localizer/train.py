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

def train_localizer(model, optimizer, data_loader, epochs=100):
    losses = []
    model = model.to(device)
    for e in range(epochs):
        for train_batch, labels_batch in data_loader:
            x = train_batch.to(device=device, dtype=dtype)
            y = labels_batch.to(device=device, dtype=dtype)
            y_hat = model(x)

            loss = calculate_loss(y_hat, y)
            if e % 10 == 0:
                print("Loss = ", loss.item())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses, y_hat, y

def calculate_loss(y_hat, y):
    '''
        y_hat 550
        y - 550 {5x5x2x11}
    '''
    loss = torch.nn.MSELoss()
    return loss(y_hat, y.view(y.shape[0], -1))

def calculate_map(y_hat, y):
    pass

def calculate_iou(y_hat, y):
    pass
