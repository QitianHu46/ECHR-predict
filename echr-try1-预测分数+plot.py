import pandas as pd 
import torch
import numpy as np 
import tqdm, pickle 
from echr_utils import  *

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

from torch.utils.data import DataLoader
# f = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/test.pickle', 'rb') 
# test_dataset = pickle.load(f)
# dt_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda') 
    print('用GPU！！！')
else: 
    device = torch.device('cpu')
    print('在用CPU。。。')

model = torch.load('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/saved_model/test2')
model.to(device)
model.eval()
softmax1 = torch.nn.Softmax(dim=1)

# # 搞数据
d = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-dev.csv')
d['l'] = d.TEXT.apply(len)
d = d[d.l < 45000]

from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt

def draw_one(i = 3):
    all_sent = sent_tokenize(d.iloc[i]['TEXT'])
    batch = tokenizer(all_sent, truncation=True, padding=True)
    input_ids = torch.as_tensor(batch['input_ids']).to(device)
    attention_mask = torch.as_tensor(batch['attention_mask']).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    res = softmax1(outputs['logits'].to('cpu'))[:,1] # 这个可以是预测 violate 的一个分数，在 0 ~ 1 之间

    plt.hist(res.detach().numpy())
    plt.title('Violate = '+ str(d.iloc[i].violate)+ '; len = '+ str(len(all_sent)))
    plt.savefig('/net/scratch/jasonhu/legal_dec-sum/ECHR-predict/pics/test{}.png'.format(i))
    plt.close(fig='all')
    # model = torch.load('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/saved_model/test2'); model.to(device); model.eval()

for i in tqdm.tqdm(range(d.shape[0])):
    draw_one(i)
    
