import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel


def split_list(input_list, max_length=200):
    num_full_sublists = len(input_list) // max_length
    minibatch = [input_list[i * max_length:(i + 1) * max_length] for i in range(num_full_sublists)]
    if len(input_list) % max_length != 0:
        minibatch.append(input_list[num_full_sublists * max_length:])
    return minibatch

class ItemsDatasetEmbedded(Dataset):
    def __init__(self, item_data, review_data):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert.eval()
        merged_data = pd.merge(review_data, item_data[['gmap_id', 'description']], on='gmap_id', how='left')
        self.item_data = {}
        
        for idx, row in merged_data.iterrows():
            gmap_id = row['gmap_id']
            user_id = row['user_id']
            text = row['text']
            rating = row['rating']
            description = row['description']
            if gmap_id not in self.item_data:
                self.item_data[gmap_id] = {
                    'user_ids': [],
                    'texts': [],
                    'ratings': [],
                    'description': description
                }
            self.item_data[gmap_id]['user_ids'].append(user_id)
            self.item_data[gmap_id]['texts'].append(text)
            self.item_data[gmap_id]['ratings'].append(rating)

        print('Preprocessing data...')
        for gmap_id, data in tqdm(self.item_data.items()):
            description_tokenized = self.tokenizer(data['description'], return_tensors="pt", padding='max_length', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                data['description'] = self.bert(**description_tokenized)[0][:,0,:].to('cpu')
            torch.cuda.empty_cache()
            texts = data['texts']
            minibatch = split_list(texts, 200)
            texts_embedded = []
            for batch in minibatch:
                texts_tokenized = self.tokenizer(batch, return_tensors="pt", padding='longest', truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    embedding = self.bert(**texts_tokenized)[0][:,0,:]
                texts_embedded.append(embedding.to('cpu'))
                del embedding
                del texts_tokenized
                torch.cuda.empty_cache()
            data['texts'] = torch.cat(texts_embedded, dim=0)
            torch.cuda.empty_cache()
        print('Data initialized and preprocessed.')

    def __len__(self):
        return len(self.item_data)

    def __getitem__(self, idx):
        gmap_id = list(self.item_data.keys())[idx]
        user_ids = torch.tensor(self.item_data[gmap_id]['user_ids'], dtype=torch.int64)
        ratings = torch.tensor(self.item_data[gmap_id]['ratings'], dtype=torch.float32)
        description = self.item_data[gmap_id]['description']
        texts = self.item_data[gmap_id]['texts']
        return gmap_id, user_ids, description, texts, ratings



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))


    review_30 = pd.read_csv('review_30.csv', index_col=0)
    item_30 = pd.read_csv('item_30.csv', index_col=0)
    review_train = pd.read_csv('review_train.csv', index_col=0)
    review_test = pd.read_csv('review_test.csv', index_col=0)
    assert review_30.isnull().sum().sum() == 0
    assert item_30.isnull().sum().sum() == 0
    assert review_train.isnull().sum().sum() == 0
    assert review_test.isnull().sum().sum() == 0

    torch.cuda.empty_cache()
    embedded_dataset = ItemsDatasetEmbedded(item_30, review_train)
    torch.save(embedded_dataset, 'dataset/embedded_dataset.pt')