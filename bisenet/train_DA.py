from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from model.Bisenet import BiSeNet
from model.discriminator import FCDiscriminator
from torchvision import models
from torch.backends import cudnn
from Liuloss import DiceLoss,CrossEntropy2d,get_target_tensor
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import torch.nn as nn
CUDA_LAUNCH_BLOCKING=1
from Cityscapes.CityscapesLiu import Cityscape
from GTA5.GTA5 import GTA5

#prepare dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transformtrain = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
targetdataset = Cityscape('Cityscapes/', transforms=transformtrain,)
sourcedataset = GTA5('GTA5/', transforms=transformtrain,)
sourcedata_loader = torch.utils.data.DataLoader(
        sourcedataset, batch_size=1, shuffle=True, num_workers=0)
targetdata_loader=torch.utils.data.DataLoader(
        targetdataset, batch_size=1, shuffle=True, num_workers=0)

source_train_loader_it = iter(sourcedata_loader)
target_train_loader_it = iter(targetdata_loader)
#prepare network
model=BiSeNet(19)
model_d=FCDiscriminator(19)
model=model.to(device)
model_d=model_d.to(device)

#prepare optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
optimizer_d = torch.optim.SGD(model_d.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d,
                                                   step_size=3,
                                                   gamma=0.1)
#prepare loss function
criterion_seg= CrossEntropy2d()
criterion_d=BCEWithLogitsLoss()

# labels for adversarial training
source_label = 0
target_label = 1

#parameters

NUM_EPOCHS = 10

cudnn.benchmark # Calling this optimizes runtime
LOG_FREQUENCY = 50
discriminator_source_loss=[]
discriminator_target_loss=[]
totalstep=[]
current_step = 0
lambda_adv=0.15
# Starting train
for epoch in range(NUM_EPOCHS):
    print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, NUM_EPOCHS, scheduler.get_lr()))
    i_iter=0
    for i_iter in range(len(source_train_loader_it)):
        # Set model to train
        model.train()
        model_d.train()
        # Zero-grad the optimizers
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        # Get source/target images and labels and move them to GPUs
        try:
            source_images, source_labels = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(sourcedata_loader)
            source_images, source_labels = next(source_train_loader_it)

        try:
            target_images,_ = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(targetdata_loader)
            target_images,_ = next(target_train_loader_it)

        source_images = source_images.to(device, dtype=torch.float32)
        target_images= target_images.to(device, dtype=torch.float32)
        source_labels = source_labels.to(device, dtype=torch.long)
        # train don't accumulate gradients in discriminator
        for param in model_d.parameters():
            #discriminator 的参数不会更新
            param.requires_grad = False
            pass
        # Train Source
        #print(source_images.shape)
        spreds, spreds_sup1, spreds_sup2 = model(source_images)
        loss1 = criterion_seg(spreds, source_labels)
        loss2 = criterion_seg(spreds_sup1, source_labels)
        loss3 = criterion_seg(spreds_sup2, source_labels)
        loss_seg_source = loss1 + loss2 + loss3
        # 用source更新generator的参数
        loss_seg_source.backward()
        # Train Target
        tpreds, tpreds_sup1, tpreds_sup2 = model(target_images)
        #fool the discriminator
        d_output = model_d(F.softmax(tpreds, dim=1))
        #实际上预测与损失来自于target，但我们骗generator损失来自于source
        loss_fool = criterion_d(d_output,
                                get_target_tensor(d_output, "source").to(device, dtype=torch.float))
        loss_target = loss_fool * lambda_adv
        #用target更新generator的参数
        loss_target.backward()

        # TRAIN DISCRIMINATOR
        for param in model_d.parameters():
            param.requires_grad = True
            pass
        source_predictions = spreds.detach()     #detach：当我们进行反向传播时调用到此就会停止，不能再继续传播，实际的意义为我们只更改discriminator的参数
        target_predictions = tpreds.detach()

        d_output_source = model_d(F.softmax(source_predictions, dim=1))
        target_tensor = get_target_tensor(d_output_source, "source")
        source_d_loss = criterion_d(d_output_source, target_tensor.to(device, dtype=torch.float)) / 2
        source_d_loss.backward()

        d_output_target = model_d(F.softmax(target_predictions, dim=1))
        target_tensor = get_target_tensor(d_output_target, "target")
        target_d_loss = criterion_d(d_output_target, target_tensor.to(device, dtype=torch.float)) / 2
        target_d_loss.backward()

        discriminator_source_loss.append(source_d_loss.item())
        discriminator_target_loss.append(target_d_loss.item())

        current_step += 1
        totalstep.append(current_step)


        if i_iter % LOG_FREQUENCY == 0:
            print('Step {},source_d_Loss {},target_d_loss {}'.format((i_iter),source_d_loss.item(),target_d_loss.item()))




    scheduler.step()
    scheduler_d.step()
    pass
torch.save(model,'modelad')
torch.save(model_d,'modelad_d')

plt.plot(totalstep,discriminator_source_loss)
plt.show()
plt.plot(totalstep,discriminator_target_loss)
plt.show()



























'''
for epoch in range(NUM_EPOCHS):
  print('Starting epoch {}/{}, LR = {}'.format(epoch+1, NUM_EPOCHS, scheduler.get_lr()))

  # Iterate over the dataset
  for images, labels in traindata_loader:
    # Bring data over the device of choice
    images = images.to(device,dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
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
torch.save(net,'net50256.pkl')




plt.plot(totalstep,totalloss)
plt.show()
'''



