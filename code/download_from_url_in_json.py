#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, multiprocessing, urllib3, csv
from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json


# In[2]:


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# In[4]:


def ParseData(data_file):
    assemble_list = []
    j = json.load(open(data_file))
    images = j['data']
    for item in images:
        assembleId = item['assembleId']
        assembleContent = item['assembleContent']
        assembleText = item['assembleText']
        assembleScratchTime = item['assembleScratchTime']
        facetId = item['facetId']
        sourceId = item['sourceId']
        domainId = item['domainId']
        url = item['url']
        type_ = item['type']
        assemble_list.append((assembleId, assembleContent, assembleText, assembleScratchTime, facetId, sourceId, domainId, url, type_))
    return assemble_list


# In[5]:


def DownloadImage(assemble):
    out_dir = sys.argv[2]
    (assembleId, assembleContent, assembleText, assembleScratchTime, facetId, sourceId, domainId, url, type_) = assemble
    assembleId = str(assembleId)
    filename = os.path.join(out_dir, assembleId+'.jpg') #扩展名怎么办呢？
 
    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return
 
    try:
        #print('Trying to get %s.' % url)
        http = urllib3.PoolManager(timeout=10.0)
        response = http.request('GET', url)
        image_data = response.data
    except:
        print('Warning: Could not download image %s from %s' % (assembleId, url))
        return
 
    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s %s' % (assembleId, url))
        return
 
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % assembleId)
        return
 
    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image %s' % filename)
        return


# In[6]:


def Run():
    
    if len(sys.argv) != 3:
        print('Syntax: %s <train|val|test.json.json> <output_dir/>' % sys.argv[0])
        
        sys.exit(1)  
        """
         exit(0)：无错误退出
         exit(1)：有错误退出
         """
 
    data_file, out_dir = sys.argv[1:]
    
 
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
 
    assemble_list = ParseData(data_file)
    print("222")
    pool = multiprocessing.Pool(processes=12)
 
    with tqdm(total=len(assemble_list)) as t:
         for _ in pool.imap_unordered(DownloadImage, assemble_list):
            t.update(1)


# In[7]:


if __name__ == '__main__':
    Run()

