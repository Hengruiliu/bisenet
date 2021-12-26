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
        image = image.resize((512,512), Image.BILINEAR)
        label = label.resize((512,512), Image.NEAREST)
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



















'''
#encode the label
def encode_segmap(mask, mapping, ignore_index):
    label_copy = ignore_index * np.ones(mask.shape, dtype=np.float32)
    for k, v in mapping:
        label_copy[mask == k] = v

    return label_copy
#
info = json.load(open('info.json','r'))      #加载json数据
class_mapping = info['label2train']           #取label2train数据
ignore_index=255
#open image and label
images=os.listdir('train/')
labels=os.listdir('train_labels/')
imagee=[]
labell=[]
#__getitem__
for i in range(len(images)):
    image=Image.open('train/'+images[i])
    label=Image.open('labels/'+labels[i])
    # convert into numpy array
    image = np.asarray(image, np.float32)
    image=torch.from_numpy(image)
    label = np.asarray(label, np.float32)
    # remap the semantic label
    label = encode_segmap(label,class_mapping, ignore_index)
    label=torch.from_numpy(label)
    imagee.append(image)
    labell.append(label)
    pass
'''







