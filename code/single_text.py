#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import torchvision.models as models
import torch
import torch.nn as nn


# In[2]:


import os
path = '../数据结构人工/树/'
features_dir = '../数据结构人工text_fea'
text_dir = path + 'text'
filename = os.path.join(text_dir, '2861663.txt')
#print(filename)
f = open(filename, 'r')
text = f.read()
print(text)


# In[3]:


class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('bert-base-chinese')
        self.textExtractor = BertModel.from_pretrained(
            'bert-base-chinese', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size   #embedding_dim应该是模型截断处输出的维度

        self.fc = nn.Linear(embedding_dim, code_length)  
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features


# In[4]:


textNet = TextNet(code_length=32)
textNet.cuda()   #GPU上运行


# In[4]:


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

#texts = ["[CLS] Who was Jim Henson ? [SEP]", "[CLS] Jim Henson was a puppeteer [SEP]"]
tokens, segments, input_masks = [], [], []
#for text in texts:
tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
tokens.append(indexed_tokens)
segments.append([0] * len(indexed_tokens))
input_masks.append([1] * len(indexed_tokens))

max_len = max([len(single) for single in tokens])  # 最大的句子长度

for j in range(len(tokens)):
    padding = [0] * (max_len - len(tokens[j]))
    tokens[j] += padding
    segments[j] += padding
    input_masks[j] += padding


# In[5]:


tokens_tensor = torch.tensor(tokens)
segments_tensors = torch.tensor(segments)
input_masks_tensors = torch.tensor(input_masks)


tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
input_masks_tensors = input_masks_tensors.to('cuda')


# In[6]:


text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)


# In[7]:


print(text_hashCodes)

