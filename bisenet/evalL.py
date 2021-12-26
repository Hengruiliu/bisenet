from torch.utils.data import DataLoader
import torch
import tqdm
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from model.Bisenet import BiSeNet
from torchvision import models
from torch.backends import cudnn
from Liuloss import DiceLoss,CrossEntropy2d
import torch.nn as nn
CUDA_LAUNCH_BLOCKING=1
from Cityscapes.CityscapevalL import Cityscape
net2=torch.load('net50.pkl')
net2.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device=torch.device('cpu')
transformtrain = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dataset = Cityscape('Cityscapes/', transforms=transformtrain)   # get_transform(train=True)

val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0,)
num_classes=18
miou=[]
step=[]
i=1
NUM_EPOCHS=10
for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
                for images, labels in val_loader:
                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)
                        #print('labels',labels.shape)
                        output = net2(images)


                        output = output.cpu().numpy()
                        output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
                        #print(output.shape)
                        labels = labels.cpu().numpy()
                        #
                        intersection = np.logical_and(labels, output)
                        union = np.logical_or(labels, output)
                        iou_score = np.sum(intersection) / np.sum(union)
                        print(iou_score)
                        miou.append(iou_score)
                        step.append(i)
                        i=i+1
                        #
                        pass
                pass
        pass



print('miou: ',np.mean(miou))
plt.plot(step,miou)
plt.show()
