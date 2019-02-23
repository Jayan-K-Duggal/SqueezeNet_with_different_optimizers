#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 23:05:06 2019

@author: iot
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import numpy as np
import torchvision
import torchvision.transforms as transforms
import math
import sys
import time
import os
from livelossplot import PlotLosses

lrt           = 0.01                                                                                          
start_epoch  = 1
num_epochs   = 200
batch_size   = 128
best_acc     = 0
criterion = nn.CrossEntropyLoss()

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./train_data', transform=transform_train, train=True, download=True)
test_dataset  = torchvision.datasets.CIFAR10(root='./test_data', transform=transform_test, train=False, download=True)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=80, num_workers=8, shuffle=False)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=10):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    return model        

net = squeezenet1_0()
if is_use_cuda:
    net.to(device)
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
tmp = torch.randn(1, 3, 32, 32)
y   = net(tmp)
#print(y, type(y), y.size())

def train(epoch):
    global train_loss
    global train_correct
    global train_total
    global optimizer
    net.train()
    train_loss = 0
    train_correct    = 0
    train_total      = 0
    optimizer  = optim.Adagrad(net.parameters(), lrt, weight_decay=5e-4)
    
    print('Sqnet_1x_v1.0_Adagrad Training Epoch: #%d, LR: %.4f'%(epoch, lrt))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        train_total      += labels.size(0)
        train_correct    += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Tr_Acc: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), train_correct / train_total))
        sys.stdout.flush()
          
def test(epoch):
    global test_loss
    global test_correct
    global test_total
    global best_acc
    net.eval()
    test_loss = 0
    test_correct   = 0
    test_total     = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        
        test_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        test_total      += labels.size(0)
        test_correct    += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Te_Acc: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(test_dataset) // 80, 
                          test_loss / (100 * (idx + 1)), test_correct / test_total))
        sys.stdout.flush()
        
    if test_correct / test_total > best_acc:
        print()
        print('Saving Model...')
        state = {
             'net': net.module if is_use_cuda else net,
             'net_state_dict': net.state_dict(),
#             'acc': test_correct / test_total,
#             'optimizer_state_dict': optimizer.state_dict()
        }
        if not os.path.isdir('./checkpoint/Sqnet_1x_v1.0_Adagrad'):
            os.makedirs('./checkpoint/Sqnet_1x_v1.0_Adagrad')
        torch.save(state, './checkpoint/Sqnet_1x_v1.0_Adagrad/Sqnet_1x_v1.0_Adagrad_Cifar10.ckpt')
        best_acc = test_correct / test_total 
#        checkpoint = torch.load('./checkpoint/Sqnet_1x_v1.0/Sqnet_1x_v1.0_Cifar10.ckpt')
#        net.load_state_dict(checkpoint['net_state_dict'])
#        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
liveloss = PlotLosses()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()
    train(_epoch)
    print()
    test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))
    best_cost = end_time - start_time
    if end_time - start_time < best_cost:
        best_cost = end_time - start_time  
              
    liveloss.update({
        'log loss': train_loss,
        'val_log loss': test_loss,
        'accuracy': train_correct,
        'val_accuracy': test_correct
    })
    liveloss.draw()
print('Best Cost: %ds' % (best_cost))    
print('Best Acc: %.4f percent' % (best_acc * 100))


    
    