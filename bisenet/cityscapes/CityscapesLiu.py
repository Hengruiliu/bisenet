import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import json



def encode_segmap(mask, mapping, ignore_index):
    label_copy = ignore_index * np.ones(mask.shape, dtype=np.float32)
    for k, v in mapping:
        label_copy[mask == k] = v

    return label_copy

class Cityscape(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.ignore_index=225
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.info = json.load(open('Cityscapes/info.json', 'r'))
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "train_labels"))))
        self.class_mapping = self.info['label2train']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        label_path = os.path.join(self.root, "train_labels", self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        label=Image.open(label_path)
        image = image.resize((512,256), Image.BILINEAR)
        label = label.resize((512,256), Image.NEAREST)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label = encode_segmap(label, self.class_mapping, self.ignore_index)

        image = image.transpose((2, 0, 1))
        #image=torch.from_numpy(image)
        #label=torch.from_numpy(label)
        '''
        if self.transforms is not None:
            image, label = self.transforms(image,label)
            pass
            '''
        return image, label



























