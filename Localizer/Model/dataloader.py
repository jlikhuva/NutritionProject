import os
import sys
import torch
import numpy as np
from PIL import Image
from itertools import chain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
sys.path.append("..")
from Shared import utils


class NutritionDataset(Dataset):
    def __init__(
        self, image_dir, bounding_boxes_path,
        data_path, split='train', shrink_factor=(4, 4),
        debug=True, mean_normalize=False,
        mean_path=None
    ):
        self.cur_split_images = np.load(data_path).item()[split]
        if debug:
            self.images = [os.path.join(image_dir, f) for f in self.cur_split_images[:20]]
        else:
            self.images = [os.path.join(image_dir, f) for f in self.cur_split_images[:]]
        self.bounding_boxes = np.load(bounding_boxes_path).item()
        self.shrink_factor = shrink_factor
        self.mean_normalize = mean_normalize
        self.mean_path = mean_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = image.resize(
            ((1080//self.shrink_factor[0]), (1920//self.shrink_factor[1])),
            resample=Image.BILINEAR
        )
        y = self._create_yolo_target_tensor(idx)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(image)
        if self.mean_normalize:
            norm_t = transforms.Compose([utils.SubtructMeanImage(self.mean_path)])
            image = norm_t(image)
        return image, y

    def _create_yolo_target_tensor(self, idx):
        nutrition, ingridients, _ = self.bounding_boxes[self.cur_split_images[idx]]
        nutrition, ingridients = linearly_scale_bounding_boxes(
            nutrition, ingridients, self.shrink_factor[0],
            self.shrink_factor[1]
        )
        return create_target_tensor(
            nutrition, ingridients,
            1080//self.shrink_factor[0],
            1920//self.shrink_factor[1]
        )

def get_bounding_rect(coords):
    '''
    Transform an arbitrarily shaped quadrilateral into a
    the smallest Rectangle that it can fit in.
    '''
    x_max, x_min = max(coords[::2]), min(coords[::2])
    y_max, y_min = max(coords[1:][::2]), min(coords[1:][::2])
    center_x = (x_min + x_max)/2; center_y = (y_min + y_max)/2
    width = (x_max - x_min); height = (y_max - y_min)
    return np.array([center_x, center_y, width, height]), (center_x, center_y)

def create_target_tensor(nutrition, ingridients, w, h, S=3, K=7,  B=2):
    cell_w = w//S ;cell_h = h//S
    if K == 11:
        ingr_center = get_bbox_center(ingridients)
        nutr_center = get_bbox_center(nutrition)

    ingridients = [(t[0]/float(w), t[1]/float(h)) for t in ingridients]
    nutrition = [(t[0]/float(w), t[1]/float(h)) for t in nutrition]
    target_tensor = np.zeros((S, S, B, K), dtype=np.float32)
    ingr_t = np.array(
        list(chain.from_iterable(ingridients))
    )
    nutr_t = np.array(
        list(chain.from_iterable(nutrition))
    )
    if K == 7:
        ingr_t, ingr_center = get_bounding_rect(ingr_t)
        nutr_t, nutr_center = get_bounding_rect(nutr_t)


    ing_found, nut_found = False, False
    for i in range(S):
        for j in range(S):
            start = (i*cell_w, j*cell_h)
            nutr_in_cell = center_in_cell(nutr_center, start, cell_w, cell_h)
            ingr_in_cell = center_in_cell(ingr_center, start, cell_w, cell_h)
            if nutr_in_cell and not nut_found:
                nut_found = True

                target_tensor[i, j, 0, 0] = 1
                target_tensor[i, j, 0, 1:-2] = nutr_t
                target_tensor[i, j, 0, -2] = 1
            if ingr_in_cell and not ing_found:
                ing_found = True

                target_tensor[i, j, 1, 0] = 1
                target_tensor[i, j, 1, 1:-2] = ingr_t
                target_tensor[i, j, 1, -1] = 1
    # print (target_tensor.shape)
    return torch.FloatTensor(target_tensor)

def get_bbox_center(coords):
    x = sum([t[0] for t in coords])//4
    y = sum([t[1] for t in coords])//4
    return (x, y)

def center_in_cell(center, cell_start, cell_w, cell_h):
    cell_start_x = cell_start[0]; cell_end_x = cell_start_x + cell_w
    cell_start_y = cell_start[1]; cell_end_y = cell_start_y + cell_h
    if (center[0] < cell_start_x) or (center[0] > cell_end_x):
        return False
    if (center[1] < cell_start_y) or (center[1] > cell_end_y):
        return False
    return True

def linearly_scale_bounding_boxes(
  nutrition, ingridients,
  shrink_factor_w, shrink_factor_h
):
    nutrition_coords = [(
            int(nutrition[i])/shrink_factor_w, int(nutrition[i+1])/shrink_factor_h
        ) for i in range(0, len(nutrition), 2)]
    ingridients_coords = [(
            int(ingridients[i])/shrink_factor_w, int(ingridients[i+1])/shrink_factor_h
        ) for i in range(0, len(ingridients), 2)]
    return (nutrition_coords, ingridients_coords)
