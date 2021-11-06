do_data = True  ; save_data = False
do_training = True 
do_checking = False  
print('do_data = ', do_data)
print('do_training = ', do_training)

import pandas as pd 
import torch
import numpy as np 
import tqdm 
torch.cuda.empty_cache()
import pickle 

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

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('用GPU！！！')
else: 
    device = torch.device('cpu')
    print('在用CPU。。。')


if do_data:

    from transformers import LongformerTokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    
    def load_tokenize_save(data_path, save_file_name, save_data=save_data):
        te = pd.read_csv(data_path)[:100]
        test_texts, test_labels = te.TEXT.tolist(), te.violate.tolist()
        del te 
        val_encodings = tokenizer(test_texts, truncation=True, 
            padding='max_length', max_length=4096)
            # padding='max_length', max_length=512)
        val_dataset = IMDbDataset(val_encodings, test_labels)
        if save_data:
            filehandler = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/'+save_file_name, 'wb') 
            pickle.dump(val_dataset, filehandler)
            print(save_file_name + ' dumped')
        return val_dataset
    
    test_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-test.csv', save_file_name='test.pickle', save_data=save_data)
    # train_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-train.csv', save_file_name='train.pickle', save_data=save_data)
    val_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/ecthr-dev.csv', save_file_name='dev.pickle', save_data=save_data)

    print('dataset fin')
else:
    f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/train.pickle', 'rb') 
    train_dataset = pickle.load(f)
    f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/test.pickle', 'rb') 
    test_dataset = pickle.load(f)
    f = open('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/datasets/dev.pickle', 'rb') 
    val_dataset = pickle.load(f)

############################################################
if do_training:
    from transformers import Trainer, TrainingArguments
    num_epochs = 1
    from transformers import LongformerForSequenceClassification, AdamW
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096')
    # model.half()
    model.to(device)
    
    from datasets import load_metric
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=num_epochs,              # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
        per_device_eval_batch_size=1,   # batch size for evaluation
        warmup_steps=100,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        fp16=True
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        # train_dataset=train_dataset,         # training dataset
        train_dataset=val_dataset,  
        eval_dataset=test_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    # print('第一次 evaluate !!')
    # res = trainer.evaluate()
    # print(res)
    trainer.train()
    print('第二次 evaluate !!')
    res = trainer.evaluate()
    print(res)

    torch.save(model, '/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/Longformer1')


#################################################
if do_checking:
    from torch.utils.data import DataLoader
    model = torch.load('/net/scratch/jasonhu/legal_dec-sum/ECHR_Dataset/saved_model/Longformer1')
    train_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    for epoch in range(3):
        for batch in tqdm.tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    # model(test_dataset[0]['input_ids'], test_dataset[0]['attention_mask'], test_dataset[0]['labels'] )
    # model(input_ids, attention_mask=attention_mask, labels=labels)

    # model(test_dataset[0]['input_ids'], attention_mask=test_dataset[0]['attention_mask']*0, labels=test_dataset[0]['labels'] )

    # model(test_dataset[0]['input_ids'])

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



