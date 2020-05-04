#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:11:24 2020

@author: wangconghao
"""

# -*- coding: utf-8 -*-
 
import os.path
 
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable  
 
import numpy as np
from PIL import Image  
 
features_dir = '/Users/wangconghao/Documents/毕业设计/数据结构人工/2-3树'
 
img_path = "/Users/wangconghao/Documents/毕业设计/数据结构人工/2-3树/image/2852662_0.png"
file_name = img_path.split('/')[-1]
feature_path = os.path.join(features_dir, file_name + '.txt')
 
 
transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()    ]
)
 
img = Image.open(img_path).convert('RGB')
img1 = transform1(img)
 
#resnet18 = models.resnet18(pretrained = True)
resnet50_feature_extractor = models.resnet50(pretrained = True)
resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)
 
for param in resnet50_feature_extractor.parameters():
    param.requires_grad = False
#resnet152 = models.resnet152(pretrained = True)
#densenet201 = models.densenet201(pretrained = True) 
x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#y1 = resnet18(x)
y = resnet50_feature_extractor(x)
y = y.data.numpy()
np.savetxt(feature_path, y, delimiter=',')
#y3 = resnet152(x)
#y4 = densenet201(x)
 
y_ = np.loadtxt(feature_path, delimiter=',').reshape(1, 2048)
