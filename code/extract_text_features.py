#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np

model_name = 'bert-base-multilingual-cased'


# In[2]:


import os, glob
#data_dir = '/Users/wangconghao/Documents/毕业设计/数据结构人工'
data_dir = '~/wangch/数据结构人工'
#features_dir = '/Users/wangconghao/Documents/毕业设计/数据结构人工features'
features_dir = '~/wangch/数据结构人工/feautures'


# In[3]:


def get_filelist(path):
    extention = 'txt'
    file_list = []
    sub_dirs = [x[0] for x in os.walk(path)]
    sub_dirs = sub_dirs[1:]
    for sub_dir in sub_dirs:
        file_glob = os.path.join(sub_dir, '*.' + extention)
        file_list.extend(glob.glob(file_glob))   #glob文件名模式匹配
    return file_list
    
FL = get_filelist(data_dir)


# In[4]:


class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        # model_name = 'bert-base-multilingual-cased'
        modelConfig = BertConfig.from_pretrained(model_name)
        self.textExtractor = BertModel.from_pretrained(
            model_name, config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size   #embedding_dim是模型截断处输出的维度
        self.fc = nn.Linear(embedding_dim, code_length)  # code_length是特征维度
        
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features

textNet = TextNet(code_length=32)
#textNet.cuda()   #GPU上运行


# In[7]:


def extract_features(file_list, model):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.do_lower_case = False 
    for filename in file_list:
        if filename[-17:] == 'assemble_list.txt':
            continue
        print(filename)
        
        fea_subdir = os.path.join(features_dir, filename[16:-12] + '/txt_fea/')
        if os.path.exists(fea_subdir) is False:
            os.makedirs(fea_subdir)
        fea_file = os.path.join(fea_subdir, x_path[-11:-4] + '.txt')
        
        with open (filename) as f:
            text = f.read()
            
        if text == '\xa0' or text == '\xa0\xa0':
            continue
            
        tokens, segments, input_masks = [], [], []
        tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
#         print(tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表

        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度
        
        if max_len > 500:
        continue

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        '''
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        input_masks_tensors = input_masks_tensors.to('cuda')
        '''
        
        try:
            text_hashCodes = model(tokens_tensor, segments_tensors, input_masks_tensors)
            y = text_hashCodes.data.numpy()
            np.savetxt(fea_file, y, delimiter = ',')
#             print(y)
        except:
            print("Tokens are too long!")


# In[8]:


extract_features(FL, textNet)


# In[ ]:




