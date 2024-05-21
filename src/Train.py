import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
import numpy as np
import time
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
# from MAA import VGGNet
from time import time
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from torch.optim.lr_scheduler import CosineAnnealingLR
# import data_prepare
# from MAANN import MAA
# from FloatMAA import MAA
from QSNN import MAA
# train_loader = data_prepare.train_loader
# val_loader = data_prepare.val_loader
# test_loader = data_prepare.test_loader

import data_deal
train_loader = data_deal.train_loader 
val_loader = data_deal.val_loader
test_loader =data_deal.test_loader

torch.cuda.set_device(2)    

cudnn.benchmark = True
cudnn.deterministic = True

def angular_distance_compute(label,pred):

    mae=180-torch.abs(torch.abs(label-pred)-180)
    return torch.sum(mae)/mae.shape[0]

model = MAA().cuda()
print(model)
best_acc = 100
# loss_fun = torch.nn.MSELoss()

loss_fun = torch.nn.CrossEntropyLoss().cuda()
# optimer = torch.optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
optimer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)

scheduler = CosineAnnealingLR(optimer, T_max=150, eta_min=0)

model_save_name = 'snn-FWJA_rb50_4_Local.pth'
train_mae_list = []
test_mae_list = []
valid_mae_list = []

if __name__ == '__main__':
    sta = time()

    for i in range(150):
        loss_ce_all = 0
        start_time = time()
        model.train()
        running_mae = 0
        print("epoch:{}".format(i))
        for step, (imgs, target) in enumerate(train_loader,1):
            imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
            # imgs = imgs.unsqueeze(1).repeat(1,4,1,1,1)
            output = model(imgs)
            # loss = loss_fun(output,target)
            loss = sum([loss_fun(s, target) for s in output]) / 4
            optimer.zero_grad()
            loss.backward()
            optimer.step()

            angle_pred=torch.argmax(output.mean(dim=0),1)
            ground_truth=torch.argmax(target,1)
            mae=angular_distance_compute(angle_pred,ground_truth)
            running_mae += mae.item()
            loss_ce_all += loss.item()
            functional.reset_net(model)
            if step % 100 == 0:
                print("step:{:.2f} loss_ce:{:.4f}".format(step / len(train_loader),loss.item()))
        accuracy1 = running_mae / step
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_mae = 0
            for step ,(imgs,target) in enumerate(val_loader,1):
                imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
                # imgs = imgs.unsqueeze(1).repeat(1,4,1,1,1)
                output  = model(imgs)
                
                angle_pred=torch.argmax(output.mean(dim=0),1)
                ground_truth=torch.argmax(target,1)
                
                mae=angular_distance_compute(angle_pred,ground_truth)

                val_mae += mae.item()

            accuracy2 = val_mae / step
        
            test_mae = 0
            for step,(imgs, target) in enumerate(test_loader,1):
                imgs, target = imgs.cuda(non_blocking=True), target.cuda(non_blocking=True)
                # imgs = imgs.unsqueeze(1).repeat(1,4,1,1,1)
                output  = model(imgs)

                angle_pred = torch.argmax(output.mean(dim=0),1)
                ground_truth = torch.argmax(target,1)
                
                mae = angular_distance_compute(angle_pred,ground_truth)

                test_mae += mae.item()

    
            accuracy = test_mae / step
            end_time = time()
            print("epoch:{} time:{:.0f}  loss:{:.4f} train_acc:{:.4f} val_acc:{:.4f} tets_acc:{:.4f} eta:{:.2f}".format(i,end_time - start_time,loss_ce_all,accuracy1, accuracy2,accuracy, (end_time - start_time) * (200 - i - 1) / 3600))
            train_mae_list.append(accuracy1)
            valid_mae_list.append(accuracy2)
            test_mae_list.append(accuracy)

            if accuracy < best_acc:
                best_acc = accuracy
                print("best_acc:{:.4f}".format(best_acc))
                torch.save(model.state_dict(), model_save_name)
        np.savez("result_FJWA_Local_mae.npz",train_mae=train_mae_list,valid_mae=valid_mae_list,test_mae=test_mae_list)
    end = time()
    print(end - sta)
    print("best_acc:{:.4f}".format(best_acc))

    