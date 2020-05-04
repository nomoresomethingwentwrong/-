#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:40:56 2020

@author: wangconghao
"""

import os

path = '/Users/wangconghao/Documents/毕业设计/数据结构人工/图/'
img_dir = path + 'image'
img_list = os.listdir(img_dir)
#print(img_list)

file_name = path + 'assemble_list.txt'
f = open(file_name, 'w+')

assemble_list = []
for img in img_list:
    img = img[:-6] + '\n'
    assemble_list.append(img)
    #f.writelines(img)

#print(assemble_list)
f.writelines(assemble_list)