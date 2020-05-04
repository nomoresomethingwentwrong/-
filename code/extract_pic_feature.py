#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 01:46:36 2020

@author: wangconghao
"""

# -*- coding: utf-8 -*-
import os, torch, glob
import numpy as np
from torch.autograd import Variable
from PIL import Image  
from torchvision import models, transforms
import torch.nn as nn
import shutil
data_dir = '/Users/wangconghao/Documents/毕业设计/数据结构人工'
features_dir = '/Users/wangconghao/Documents/毕业设计/数据结构人工features'
#shutil.copytree(data_dir, features_dir) #为什么要拷贝一下数据
 
 
def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()    ]
    )
    
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()
    np.savetxt(saved_path, y, delimiter=',')
    
if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png']
        
    files_list = []
    sub_dirs = [x[0] for x in os.walk(data_dir) ]   #每次walk遍历返回一个三元组，第一项是当前文件夹的地址
    sub_dirs = sub_dirs[1:]     #第一项就是data_dir，不要了
    for sub_dir in sub_dirs:
        for extention in extensions:
            file_glob = os.path.join(sub_dir, '*.' + extention)
            files_list.extend(glob.glob(file_glob)) #glob文件名模式匹配
        
#    print(files_list)
    resnet50_feature_extractor = models.resnet50(pretrained = True)
    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)   #nn.Linear(in_fea, out_fea)
    torch.nn.init.eye_(resnet50_feature_extractor.fc.weight)    #线性地导出resnet50最后一层
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False   
        
    use_gpu = torch.cuda.is_available()
 
    for x_path in files_list:
        print(x_path)
        fea_subdir = os.path.join(features_dir, x_path[41:-20] + '/features/')
        if os.path.exists(fea_subdir) is False:
            os.makedirs(fea_subdir)
        fx_path = os.path.join(fea_subdir, x_path[-13:-4] + '.txt')
#        fx_path = x_path + '.txt'
        extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)
