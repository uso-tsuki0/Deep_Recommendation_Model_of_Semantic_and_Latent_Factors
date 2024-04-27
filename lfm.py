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
import wandb
from torch import optim
from tqdm.auto import tqdm, trange
from embedded_dataset import ItemsDatasetEmbedded, split_list
from torch.nn import MSELoss


class ItemsDataset(Dataset):
    def __init__(self, reviews):
        self.item_data = {}
        for idx, row in reviews.iterrows():
            gmap_id = row['gmap_id']
            user_id = row['user_id']
            rating = row['rating']
            if gmap_id not in self.item_data:
                self.item_data[gmap_id] = {'user_ids': [], 'ratings': []}
            self.item_data[gmap_id]['user_ids'].append(user_id)
            self.item_data[gmap_id]['ratings'].append(rating)

    def __len__(self):
        return len(self.item_data)

    def __getitem__(self, idx):
        gmap_id = list(self.item_data.keys())[idx]
        user_ids = torch.tensor(self.item_data[gmap_id]['user_ids'])
        ratings = torch.tensor(self.item_data[gmap_id]['ratings'], dtype=torch.float).unsqueeze(0)
        return gmap_id, user_ids, ratings
    


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, mask):
        select_y_pred = y_pred[mask]
        select_y_true = y_true[mask]
        return nn.MSELoss()(select_y_pred, select_y_true)


class LFM(nn.Module):
    def __init__(self, n_item, n_user, n_dim, global_bias):
        super(LFM, self).__init__()
        self.user_emb = nn.Embedding(n_user, n_dim)
        self.item_emb = nn.Embedding(n_item, n_dim)
        torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.item_emb.weight)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_bias))
        
    def forward(self, item_ids, user_ids):
        I = self.item_emb(item_ids)
        U = self.user_emb(user_ids)
        b_i = self.item_bias(item_ids)
        b_u = self.user_bias(user_ids).T
        mu = self.global_bias
        return torch.matmul(I, U.T) + b_i + b_u + mu
    
def custom_collate_fn(batch):
    return batch[0]


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()

    # train test split
    review_30 = pd.read_csv('review_30.csv', index_col=0)
    item_30 = pd.read_csv('item_30.csv', index_col=0)
    review_train = pd.read_csv('review_train.csv', index_col=0)
    review_test = pd.read_csv('review_test.csv', index_col=0)

    # interaction matrix
    review_ratings = review_30[['user_id', 'gmap_id', 'rating']].copy()
    item_user_df = review_ratings.pivot_table(index='gmap_id', columns='user_id', values='rating', aggfunc='mean')
    item_user_df = item_user_df.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    mask = torch.tensor(item_user_df.notnull().to_numpy())
    item_user_matrix = torch.tensor(item_user_df.fillna(0).to_numpy()).float()
    n_item, n_user = item_user_matrix.shape
    item_all = review_ratings['gmap_id'].unique().tolist()
    user_all = review_ratings['user_id'].unique().tolist()
    review_ratings_train = review_train[['user_id', 'gmap_id', 'rating']].copy()
    item_user_df_train = review_ratings_train.pivot_table(index='gmap_id', columns='user_id', values='rating', aggfunc='mean')
    item_user_df_train = item_user_df_train.reindex(index=item_all, columns=user_all).sort_index(ascending=True).sort_index(axis=1, ascending=True)
    item_user_matrix_train = torch.tensor(item_user_df_train.fillna(0).to_numpy()).float()
    mask_train = torch.tensor(item_user_df_train.notnull().to_numpy())

    # dataset
    print('Loading data...')
    embedded_dataset = torch.load('dataset/embedded_dataset.pt')

    # training
    batch_size = 1
    embedding_size = 768
    learning_rate = 1e-4
    epochs = 15
    mask_train = mask_train.to('cpu')
    global_bias = torch.mean(item_user_matrix[mask_train]).item()

    wandb.init(project="final_proj", name="run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lfm_model = LFM(n_item, n_user, embedding_size, global_bias).to(device)
    optimizer = optim.AdamW(lfm_model.parameters(), lr=learning_rate)
    mseloss = MSELoss()
    dataloader = DataLoader(embedded_dataset, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)

    lfm_model.train()
    for epoch in trange(epochs, desc="Epoch"):
        loss_sum = 0
        for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            item_id, user_ids, _, _, ratings = data
            item_id = torch.tensor([item_id]).to(device)
            user_ids = user_ids.to(device)
            ratings = ratings.unsqueeze(0).to(device)
            y_pred = lfm_model(item_id, user_ids)
            loss = mseloss(y_pred, ratings)
            lfm_model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if step % 100 == 99:
                wandb.log({"loss": loss_sum}, step=step+epoch*len(dataloader))
                loss_sum = 0

    lfm_model.eval()
    torch.save(lfm_model.state_dict(), 'models/lfm_model.pt')
    print('Model saved.')