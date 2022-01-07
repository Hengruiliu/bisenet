from torch.utils.data import DataLoader
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
CUDA_LAUNCH_BLOCKING=1
from Cityscapes.CityscapevalL import Cityscape
import os
palette = [128, 64, 128,  # Road, 0
            244, 35, 232,  # Sidewalk, 1
            70, 70, 70,  # Building, 2
            102, 102, 156,  # Wall, 3
            190, 153, 153,  # Fence, 4
            153, 153, 153,  # pole, 5
            250, 170, 30,  # traffic light, 6
            220, 220, 0,  # traffic sign, 7
            107, 142, 35,  # vegetation, 8
            152, 251, 152,  # terrain, 9
            70, 130, 180,  # sky, 10
            220, 20, 60,  # person, 11
            255, 0, 0,  # rider, 12
            0, 0, 142,  # car, 13
            0, 0, 70,  # truck, 14
            0, 60, 100,  # bus, 15
            0, 80, 100,  # train, 16
            0, 0, 230,  # motor-bike, 17
            119, 11, 32]  # bike, 18]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
    pass
###
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


net2=torch.load('modelad')
net2.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device=torch.device('cpu')
transformtrain = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dataset = Cityscape('Cityscapes/', transforms=transformtrain)   # get_transform(train=True)

val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,)

miou=[]
step=[]
outputs=[]
i=1
NUM_EPOCHS=1
for epoch in range(NUM_EPOCHS):
        with torch.no_grad():
                for images, labels in val_loader:
                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)
                        #print('labels',labels.shape)
                        output = net2(Variable(images, volatile=True))
                        #print(output.shape)


                        output = output.cpu().numpy()

                        output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
                        outputs.append(output)

                        #print(output.shape)
                        labels = labels.cpu().numpy()
                        #
                        intersection = np.logical_and(labels, output)
                        union = np.logical_or(labels, output)
                        iou_score = np.sum(intersection) / np.sum(union)
                        #print(iou_score)
                        miou.append(iou_score)
                        step.append(i)
                        i=i+1
                        #
                        pass
                pass
        pass



print('miou: ',np.mean(miou))
print(outputs[1])
plt.plot(step,miou)
plt.show()

labeltotal=os.listdir('Cityscapes/val_labels')
for i in range(len(outputs)):
        out=outputs[i]
        #print(out.shape)
        outputs[i]=np.squeeze(out, 0)
        outputs[i] = np.asarray(outputs[i], dtype=np.uint8)
        outputs[i] = colorize_mask(outputs[i])
        outputs[i].save('Cityscapes/pseudolabels/{}'.format(labeltotal[i]))
        pass
