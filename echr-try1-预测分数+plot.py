
do_training = False 
do_checking = True 

############################################################

import pandas as pd 
import torch
import numpy as np 
import tqdm 
import echr_utils 


############################################################

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

from torch.utils.data import DataLoader
dt_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
model = torch.load('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/test2')
# optim = AdamW(model.parameters(), lr=5e-5)

if torch.cuda.is_available():
    device = torch.device('cuda') 
    print('用GPU！！！')
else: 
    torch.device('cpu')
    print('在用CPU。。。')
model.to(device)

softmax1 = torch.nn.Softmax(dim=1)

for batch in tqdm.tqdm(dt_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    softmax1(outputs['logits'].to('cpu'))[:,1] # 这个可以是预测 violate 的一个分数，在 0 ~ 1 之间


# te = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-test.csv')
# tr = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-train.csv')
# dev = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-dev.csv')
# print('datasets loaded')
# test_texts, test_labels = te.TEXT.tolist(), te.violate.tolist()
# train_texts, train_labels = tr.TEXT.tolist(), tr.violate.tolist()
# val_texts, val_labels = dev.TEXT.tolist(), dev.violate.tolist()
# del te, tr, dev 
# print('datasets deleted')
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# class IMDbDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = IMDbDataset(train_encodings, train_labels)
# val_dataset = IMDbDataset(val_encodings, val_labels)
# test_dataset = IMDbDataset(test_encodings, test_labels)


#################################################
# if False:

#     model = torch.load('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/test2')
#     model(test_dataset[0]['input_ids'], test_dataset[0]['attention_mask'], test_dataset[0]['labels'] )
#     model(input_ids, attention_mask=attention_mask, labels=labels)

#     model(test_dataset[0]['input_ids'], attention_mask=test_dataset[0]['attention_mask']*0, labels=test_dataset[0]['labels'] )

#     model(test_dataset[0]['input_ids'])


from torch.utils.data import DataLoader
dt_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
model = torch.load('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/test2')
# optim = AdamW(model.parameters(), lr=5e-5)

if torch.cuda.is_available():
    device = torch.device('cuda') 
    print('用GPU！！！')
else: 
    torch.device('cpu')
    print('在用CPU。。。')
model.to(device)

softmax1 = torch.nn.Softmax(dim=1)

for batch in tqdm.tqdm(dt_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    softmax1(outputs['logits'].to('cpu'))[:,1] # 这个可以是预测 violate 的一个分数，在 0 ~ 1 之间
