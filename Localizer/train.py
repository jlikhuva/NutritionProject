import sys
import torch
import numpy as np
sys.path.append("..")

import torch.nn.functional as F
from Shared import utils
# from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from Model.dataloader import NutritionDataset
from Model.net import LocalizerNet
from tqdm import tqdm_notebook as tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype=torch.float32
LARGE_NUMBER = 0.0027425079606473446

def train_localizer(
    model, optimizer, train_data_loader,
    dev_data_loader, epochs=1, restore=True,
    restore_path='../Data/FullData/best_model.tar',
    scheduler=None, save=True
):
    train_losses, dev_losses, train_map, dev_map = [], [], [], []
    best_loss = LARGE_NUMBER
    if torch.cuda.device_count() > 1 or restore:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if restore:
        utils.load_checkpoint(
            restore_path, model, optimizer
        )
    for e in range(epochs):
        for train_batch, labels_batch in tqdm(train_data_loader):
            x = train_batch.to(device=device, dtype=dtype)
            y = labels_batch.to(device=device, dtype=dtype)
            y_hat = model(x)
            loss = calculate_loss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            d_loss, d_map = check_perf_on_dev(dev_data_loader, model)
            map_ = calculate_map(y_hat, y)
            dev_losses.append(d_loss)
            dev_map.append(d_map)
            train_map.append(map_)
            train_losses.append(loss.item())
            if save and d_loss < best_loss:
                utils.save_checkpoint({
                    'epoch' : e+1, 'state_dict' : model.state_dict(),
                    "optim_dict" : optimizer.state_dict()
                })
                best_loss = d_loss

            if (e+1) % 5 == 0:
                print("=== Performance Check ===")
                print("\t Train Loss = ", loss.item())
                print("\t Dev Loss = ", d_loss)
                print("\t Train mAP = ", map_)
                print("\t Dev mAP = ", d_map)

        if scheduler:
            scheduler.step()
    return train_losses, dev_losses, train_map, dev_map


def check_perf_on_dev(data_loader, model):
    losses=[]; maps = []
    with torch.no_grad():
        model.eval()
        for x, y in data_loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            y_hat=model(x)
            loss=calculate_loss(y_hat, y)
            losses.append(loss.item())
            d_map = calculate_map(y_hat, y)
            maps.append(d_map)
        model.train()
    return np.mean(losses), np.mean(maps, axis=0)


def calculate_loss(y_hat, y, lambdah=0.5, S=3, B=2, K=7):
    '''
        y_hat S*S*K*B
        y S*S*K*B
        [p_object, center_x, center_y, width, height, p_c1, p_c2]
    '''
    N = y.shape[0]
    y = y.reshape(N, S*S*B, K)
    y_hat = y_hat.reshape(N, S*S*B, K)

    no_object_mask = (y[:, :, 0] != 1)
    y_no_obj = y[no_object_mask]
    y_hat_no_obj = y_hat[no_object_mask]

    object_mask = (y[:, :, 0] == 1)
    y_hat = y_hat[object_mask]
    y = y[object_mask]

    coord_loss = torch.nn.MSELoss()
    object_loss = torch.nn.MSELoss()
    no_object_loss = torch.nn.MSELoss()
    class_loss = torch.nn.MSELoss()

    a = object_loss(y_hat[:, :1], y[:, :1])
    b = coord_loss(y_hat[:, 1:K-2], y[:, 1:K-2])
    c = class_loss(y_hat[:, K-2:], y[:, K-2:])
    d =  lambdah*no_object_loss(y_hat_no_obj[:, :1], y_no_obj[:, :1])

    return a + b + c + d


def calculate_map(y_hat, y, S=3, B=2, K=7, threshold=0.9):
    '''
    y_hat is the predicted tensor
    y is the ground truth tensor.
    '''
    N=y_hat.shape[0]
    y_hat = y_hat.reshape(N, S*S*B, K)
    y = y.reshape(N, S*S*B, K)

    # y_hat[:, :, :1] = F.sigmoid(y_hat[:, :, :1])
    pred_mask = (y_hat[:, :, 0] >= threshold)
    preds = y_hat[pred_mask]

    if len(preds) > 0:
        truth = y[pred_mask]

        nutrition_preds = preds[preds[:, -2] >= threshold]
        nutrition_truth = truth[preds[:, -2] >= threshold]

        ingridient_preds = preds[preds[:, -1] >= threshold]
        ingridient_truth = truth[preds[:, -1] >= threshold]

        nutr_precision, ingr_precision = (
            get_precision(nutrition_preds, nutrition_truth),
            get_precision(ingridient_preds, ingridient_truth)
        )
    else:
        nutr_precision, ingr_precision = np.zeros(5), np.zeros(5)
    return (nutr_precision + ingr_precision) / 2


def get_precision(y_hat, y, iou_threshold=[0.5, 0.6, 0.7, 0.8, 0.9], K=7):
    N = len(y_hat); true_positives = np.zeros(len(iou_threshold))
    if N == 0: return true_positives
    for i in range(N):
        iou = calculate_iou(y_hat[i, 1:K-2], y[i, 1:K-2])
        for i in range(len(iou_threshold)):
            if iou >= iou_threshold[i]: true_positives[i] += 1
    return true_positives / N


def calculate_iou(y_pred, y_truth, h=1920/4, w=1080/4, K=7):
    '''
     y_pred and y_truth are (K-3) -tensors
     in which each number holds true universal meaning.
     And of course, these are not the droids that you're
     looking for.
    '''
    if K == 11:
        x_h, y_h = y_pred[0::2]*w, y_pred[1::2]*h
        x, y = y_truth[0::2]*w, y_truth[1::2]*h
        rect_h =  (max(x_h), min(x_h), max(y_h), min(y_h)) # (x1, x2, y1, y2)
        rect = (max(x), min(x), max(y), min(y))

        x_a, y_a = max(rect[1], rect_h[1]), max(rect[3], rect_h[3])
        x_b, y_b = min(rect[0], rect_h[0]), min(rect[2], rect_h[2])

        I = (x_b - x_a) * (y_b - y_a)
        A1 = (rect_h[0] - rect_h[1]) * (rect_h[2] - rect_h[3])
        A2 = (rect[0] - rect[1]) * (rect[2] - rect[3])
    else:
        y_pred[0::2], y_pred[1::2] = y_pred[0::2]*w, y_pred[1::2]*h
        y_truth[0::2], y_truth[1::2] = y_truth[0::2]*w, y_truth[1::2]*h
        A1 = y_pred[-2]*y_pred[-1]; A2 = y_truth[-2]*y_truth[-1]
        rect_h = (
            y_pred[0]-(y_pred[-2]/2),  # x1
            y_pred[0]+(y_pred[-2]/2),  # x2
            y_pred[1]-(y_pred[-1]/2),  # y1
            y_pred[1]+(y_pred[-1]/2)   # y2
        )
        rect = (
            y_truth[0]-(y_truth[-2]/2),  # x1
            y_truth[0]+(y_truth[-2]/2),  # x2
            y_truth[1]-(y_truth[-1]/2),  # y1
            y_truth[1]+(y_truth[-1]/2)   # y2
        )
        x_a, y_a = max(rect[0], rect_h[0]), max(rect[2], rect_h[2])
        x_b, y_b = min(rect[1], rect_h[1]), min(rect[3], rect_h[3])
        I = (x_b - x_a) * (y_b - y_a)
        # print(A1.item(), A2.item(), I.item())

    IoU = I / (A1 + A2 - I)
    # print(IoU.item())
    return IoU
