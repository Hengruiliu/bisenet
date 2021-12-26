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
from Cityscapes.CityscapesLiu import Cityscape
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device=torch.device('cpu')
transformtrain = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dataset = Cityscape('Cityscapes/', transforms=transformtrain,)   # get_transform(train=True)

data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0)
net=BiSeNet(19).eval()

params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

NUM_EPOCHS = 10
net = net.to(device) # this will bring the network to GPU if DEVICE is cuda
loss_func = CrossEntropy2d()
cudnn.benchmark # Calling this optimizes runtime
LOG_FREQUENCY = 10
totalloss=[]
totalstep=[]
current_step = 0
# Start itrating over the epochs

for epoch in range(NUM_EPOCHS):
  print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))

  # Iterate over the dataset
  for images, labels in data_loader:
    # Bring data over the device of choice
    images = images.to(device,dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    #print('label',type(labels))
    #print('label',labels.shape)
    #print('imag',type(images))
    net.train() # Sets module in training mode

    # PyTorch, by default, accumulates gradients after each backward pass
    # We need to manually set the gradients to zero before starting a new iteration


    # Forward pass to the network
    output,output_sup1, output_sup2= net(images)

    #print('output',output.shape)
    #print('labels',labels.shape)
    loss1 = loss_func(output, labels)
    loss2 = loss_func(output_sup1, labels)
    loss3 = loss_func(output_sup2, labels)
    loss = loss1 + loss2 + loss3
    optimizer.zero_grad()  # Zero-ing the gradients
    #print('output',type(outputs))
    # Compute loss based on output and ground truth


    # Log loss
    if current_step % LOG_FREQUENCY == 0:
      print('Step {}, Loss {}'.format(current_step, loss.item()))
    # Compute gradients for each layer and update weights
    loss.backward()  # backward pass: computes gradients
    optimizer.step() # update weights based on accumulated gradients

    current_step += 1
    totalstep.append(current_step)
    totalloss.append(loss.item())



  # Step the scheduler
  scheduler.step()
  pass
torch.save(net,'net10city3norm(nolr).pkl')




plt.plot(totalstep,totalloss)
plt.show()




