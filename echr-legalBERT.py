import pandas as pd 

te = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-test.csv')
tr = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-train.csv')
dev = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-dev.csv')
print('datasets loaded')
train_texts, train_labels = tr.TEXT.tolist(), tr.violate.tolist()
val_texts, val_labels = dev.TEXT.tolist(), dev.violate.tolist()
test_texts, test_labels = te.TEXT.tolist(), te.violate.tolist()
 
del te, tr, dev 
print('datasets deleted')
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print('tokenization fin')

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
num_epochs = 1
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=num_epochs,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

import numpy as np 
# from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
from datasets import load_metric
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

if torch.cuda.is_available():
    device = torch.device('cuda') 
    print('用GPU！！！')
else: 
    device = torch.device('cpu')
    print('在用CPU。。。')

model.to(device)
print('第一次 evaluate !!')
res = trainer.evaluate()
print(res)
trainer.train()
print('第二次 evaluate !!')
res = trainer.evaluate()
print(res)
torch.save(model, '/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/saved_model/legalBERT1')
