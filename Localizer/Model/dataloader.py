import os
import torch
import numpy as np
from PIL import Image
from itertools import chain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class NutritionDataset(Dataset):
    def __init__(
        self, image_dir, bounding_boxes_path,
        data_path, split='train', shrink_factor=(2, 2),
        debug=True
    ):
        self.cur_split_images = np.load(data_path).item()[split]
        if debug:
            self.images = [os.path.join(image_dir, f) for f in self.cur_split_images[:2]]
        else:
            self.images = [os.path.join(image_dir, f) for f in self.cur_split_images[:5000]]
        self.bounding_boxes = np.load(bounding_boxes_path).item()
        self.shrink_factor = shrink_factor

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
        return transform(image), y

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


def create_target_tensor(nutrition, ingridients, w, h, S=5):
    cell_w = w//S ;cell_h = h//S
    ingr_center = get_bbox_center(ingridients)
    nutr_center = get_bbox_center(nutrition)
    ingridients = [(t[0]/float(w), t[1]/float(h)) for t in ingridients]
    nutrition = [(t[0]/float(w), t[1]/float(h)) for t in nutrition]
    target_tensor = np.zeros((S, S, 2, 11), dtype=np.float32)
    ingr_t = np.array(
        list(chain.from_iterable(ingridients))
    )
    nutr_t = np.array(
        list(chain.from_iterable(nutrition))
    )
    ing_found, nut_found = False, False
    for i in range(S):
        for j in range(S):
            start = (i*cell_w, j*cell_h)
            nutr_in_cell = center_in_cell(nutr_center, start, cell_w, cell_h)
            ingr_in_cell = center_in_cell(ingr_center, start, cell_w, cell_h)
            if nutr_in_cell and not nut_found:
                nut_found = True
                # print("\t Found Nutrition")
                target_tensor[i, j, 0, 0] = 1
                target_tensor[i, j, 0, 1:-2] = nutr_t
                target_tensor[i, j, 0, -2] = 1
            if ingr_in_cell and not ing_found:
                ing_found = True
                # print("\t Found ingridients")
                target_tensor[i, j, 1, 0] = 1
                target_tensor[i, j, 1, 1:-2] = ingr_t
                target_tensor[i, j, 1, -1] = 1
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
