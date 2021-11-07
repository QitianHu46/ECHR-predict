import pandas as pd 
import torch
import numpy as np 
import tqdm, pickle 

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

def prep_ECHR_data(do_data=False, save_data = False, load_data=True, save_prefix = '',# max_length=512,
                   do_test_data = True, do_train_data = True, do_val_data = True):
    test_dataset, train_dataset, val_dataset = 0,0,0 
    if do_data:
        # from transformers import LongformerTokenizer
        # tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        def load_tokenize_save(data_path, save_file_name, save_data=save_data):
            te = pd.read_csv(data_path)
            test_texts, test_labels = te.TEXT.tolist(), te.violate.tolist()
            del te 
            val_encodings = tokenizer(test_texts, truncation=True, 
                # padding='max_length', max_length=4096)
                padding='max_length', max_length=512)
            val_dataset = IMDbDataset(val_encodings, test_labels)
            if save_data:
                filehandler = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/'+save_file_name, 'wb') 
                pickle.dump(val_dataset, filehandler)
                print(save_file_name + ' dumped')
            return val_dataset

        if do_test_data:
            test_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-test.csv', save_file_name=save_prefix+'_test.pickle', save_data=save_data)
        if do_train_data:
            train_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-train.csv', save_file_name=save_prefix+'_train.pickle', save_data=save_data)
        if do_val_data:
            val_dataset = load_tokenize_save(data_path = '/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/ecthr-dev.csv', 
            save_file_name=save_prefix+'_dev.pickle', save_data=save_data)


        # return {'test_dataset': test_dataset, 'train_dataset': train_dataset,'val_dataset': val_dataset}
    else:
        if do_train_data:
            f = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/train.pickle', 'rb') 
            train_dataset = pickle.load(f)
        if do_test_data:
            f = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/test.pickle', 'rb') 
            test_dataset = pickle.load(f)
        if do_val_data:
            f = open('/net/scratch/jasonhu/legal_dec-sum/Dataset_ECHR/datasets/dev.pickle', 'rb') 
            val_dataset = pickle.load(f)

    return test_dataset, train_dataset, val_dataset


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda') 
        print('用GPU！！！')
    else: 
        device = torch.device('cpu')
        print('在用CPU。。。')
    return device 


if __name__ == "__main__":
    prep_ECHR_data(do_data=True, save_data = True, load_data=False, save_prefix = 'DistilBert',
                   do_test_data = True, do_train_data = True, do_val_data = True)
