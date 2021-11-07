
do_training, do_checking = False, True
do_data = False 

############################################################

import pandas as pd 
import torch
import numpy as np 
import tqdm, pickle 
from echr_utils import * 

# test_dataset, train_dataset, val_dataset = prep_ECHR_data(
#     do_test_data = True, do_train_data = False, do_val_data = False)

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

    device = set_device()
    model.to(device)

    print('第一次 evaluate !!')
    res = trainer.evaluate()
    print(res)
    trainer.train()
    print('第二次 evaluate !!')
    res = trainer.evaluate()
    print(res)
    torch.save(model, '/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/saved_model/test2')
if do_checking:

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    from torch.utils.data import DataLoader
    f = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/DistilBert_test.pickle', 'rb') 
    test_dataset = pickle.load(f)
    dt_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    model = torch.load('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/saved_model/test2')

    device = set_device()
    model.to(device)

    softmax1 = torch.nn.Softmax(dim=1)

    n_errors = 0
    for batch in tqdm.tqdm(dt_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        a = softmax1(outputs['logits'].to('cpu'))[:,1] # 这个可以是预测 violate 的一个分数，在 0 ~ 1 之间
        labels = batch['labels']
        n_errors += torch.logical_xor(a > .5, labels == 1).sum()

    print(n_errors / len(test_dataset))
    # from datasets import load_metric
    # metric = load_metric("accuracy")
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)
    # from transformers import Trainer, TrainingArguments
    # num_epochs=1
    # training_args = TrainingArguments(
    #     output_dir='./results',          # output directory
    #     num_train_epochs=num_epochs,              # total number of training epochs
    #     per_device_train_batch_size=1,  # batch size per device during training
    #     per_device_eval_batch_size=1,   # batch size for evaluation
    #     warmup_steps=100,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logging_dir='./logs',            # directory for storing logs
    #     logging_steps=10,
    #     fp16=True)
    # trainer = Trainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=test_dataset,             # evaluation dataset
    #     compute_metrics=compute_metrics
    # )
    # trainer.evaluate()