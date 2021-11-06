
do_training = True 
do_checking = True 

############################################################

import pandas as pd 
import torch
import numpy as np 
import tqdm, pickle 

te = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-test.csv')
tr = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-train.csv')
dev = pd.read_csv('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-dev.csv')
print('datasets loaded')
test_texts, test_labels = te.TEXT.tolist(), te.violate.tolist()
train_texts, train_labels = tr.TEXT.tolist(), tr.violate.tolist()
val_texts, val_labels = dev.TEXT.tolist(), dev.violate.tolist()


del te, tr, dev 
print('datasets deleted')
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

test_encodings = tokenizer(test_texts, truncation=True, padding=True)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

print('tokenization fin')

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
# filehandler = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/'+save_file_name, 'wb') 
# pickle.dump(val_dataset, filehandler)


# f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/train.pickle', 'rb') 
# train_dataset = pickle.load(f)
# f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/test.pickle', 'rb') 
# test_dataset = pickle.load(f)
# f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/dev.pickle', 'rb') 
# val_dataset = pickle.load(f)

############################################################
if do_training:

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
        logging_steps=10
    )

    from transformers import DistilBertForSequenceClassification, AdamW
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
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
        torch.device('cpu')
        print('在用CPU。。。')
    model.to(device)


    print('第一次 evaluate !!')
    res = trainer.evaluate()
    print(res)
    trainer.train()
    print('第二次 evaluate !!')
    res = trainer.evaluate()
    print(res)

    torch.save(model, '/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/test2')


#################################################
if do_checking:

    model = torch.load('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/test2')
    model(test_dataset[0]['input_ids'], test_dataset[0]['attention_mask'], test_dataset[0]['labels'] )
    model(input_ids, attention_mask=attention_mask, labels=labels)

    model(test_dataset[0]['input_ids'], attention_mask=test_dataset[0]['attention_mask']*0, labels=test_dataset[0]['labels'] )

    model(test_dataset[0]['input_ids'])

# #######################  另一种方式来训练  ####################################
# from torch.utils.data import DataLoader
# train_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# # optim = AdamW(model.parameters(), lr=5e-5)

# for batch in tqdm.tqdm(train_loader):
#     # optim.zero_grad()
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)
#     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)



