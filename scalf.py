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
from lfm import ItemsDataset, MaskedMSELoss, LFM


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true, mask):
        select_y_pred = y_pred[mask]
        select_y_true = y_true[mask]
        return nn.MSELoss()(select_y_pred, select_y_true)


class CustomAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomAttentionModel, self).__init__()
        self.num_heads = num_heads
        self.wk1 = nn.Linear(embed_dim, embed_dim)
        self.wk2 = nn.Linear(embed_dim, embed_dim)
        self.wv1 = nn.Linear(embed_dim, embed_dim)
        self.wv2 = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_normal_(self.wk1.weight)
        torch.nn.init.xavier_normal_(self.wk2.weight)
        torch.nn.init.xavier_normal_(self.wv1.weight)
        torch.nn.init.xavier_normal_(self.wv2.weight)
        torch.nn.init.constant_(self.wk1.bias, 0)
        torch.nn.init.constant_(self.wk2.bias, 0)
        torch.nn.init.constant_(self.wv1.bias, 0)
        torch.nn.init.constant_(self.wv2.bias, 0)

    def forward(self, query, key1, key2):
        key1 = self.wk1(key1)
        key2 = self.wk2(key2)
        value1 = self.wv1(key1)
        value2 = self.wv2(key2)
        keys = torch.cat([key1, key2], dim=0)
        values = torch.cat([value1, value2], dim=0)
        attn_output,_ = self.multihead_attn(query, keys, values, need_weights=False)
        return attn_output


class SemanticLatent(nn.Module):
    def __init__(self, n_item, n_user, n_dim=768, n_head=12, global_bias=0.0):
        super(SemanticLatent, self).__init__()
        self.item_emb = nn.Embedding(n_item, n_dim)
        self.user_emb = nn.Embedding(n_user, n_dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_bias))
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.multihead = CustomAttentionModel(n_dim, n_head)
        self.layer_norm = nn.LayerNorm(n_dim, eps=1e-6, elementwise_affine=True)
        self.fc = nn.Linear(n_dim, 1)
        self._init_weights(item_embedding, user_embedding)
        self.register_buffer('item_deep_emb', torch.zeros(n_item, n_dim))
        self.item_deep = nn.Embedding(n_item, n_dim)
        self.item_deep.weight.requires_grad = False

    def forward(self, item_id, user_ids, desc_inputs, comment_inputs):
        I = self.item_emb(item_id)
        U = self.user_emb(user_ids)
        D = self.bert(**desc_inputs)[0][:,0,:]
        C = self.bert(**comment_inputs)[0][:,0,:]
        O = self.multihead(I, D, C)[0]
        I_deep = I + self.layer_norm(O)
        with torch.no_grad():
            self.item_deep_emb.data[item_id] = I_deep.detach()
        b_i = self.item_bias(item_id)
        b_u = self.user_bias(user_ids)
        mu = self.global_bias
        return (self.fc(I_deep * U) + b_i + b_u + mu).T


class SemanticLatentLite(nn.Module):
    def __init__(self, n_item, n_user, n_dim=768, n_head=12, global_bias=0.0, item_embedding=None, user_embedding=None):
        super(SemanticLatentLite, self).__init__()
        self.user_emb = nn.Embedding(n_user, n_dim)
        self.item_emb = nn.Embedding(n_item, n_dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_bias))
        self.multihead = CustomAttentionModel(n_dim, n_head)
        self.layer_norm = nn.LayerNorm(n_dim, eps=1e-6, elementwise_affine=True)
        self.fc = nn.Linear(n_dim, 1)
        self._init_weights(item_embedding, user_embedding)
        self.register_buffer('item_deep_emb', torch.zeros(n_item, n_dim))
        self.item_deep = nn.Embedding(n_item, n_dim)
        self.item_deep.weight.requires_grad = False

    def _init_weights(self, item_embedding, user_embedding):
        if item_embedding is not None:
            self.item_emb.weight.data.copy_(item_embedding)
        else:
            torch.nn.init.xavier_uniform_(self.item_emb.weight)
        if user_embedding is not None:
            self.user_emb.weight.data.copy_(user_embedding)
        else:
            torch.nn.init.xavier_uniform_(self.user_emb.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, item_id, user_ids, desc_inputs, comment_inputs):
        I = self.item_emb(item_id)
        U = self.user_emb(user_ids)
        D = desc_inputs
        C = comment_inputs
        O = self.multihead(I, D, C)[0]
        I_deep = I + self.layer_norm(O)
        with torch.no_grad():
            self.item_deep_emb.data[item_id] = I_deep.detach()
        b_i = self.item_bias(item_id)
        b_u = self.user_bias(user_ids)
        mu = self.global_bias
        return (self.fc(I_deep * U) + b_i + b_u + mu).T
    
    def storage(self):
        self.item_deep.weight.copy_(self.item_deep_emb)
    
    def predict(self, item_ids, user_ids, batch_size=1):
        result = []
        with torch.no_grad():
            for item_id_batch in item_ids.split(batch_size):
                I_deep = self.item_deep(item_id_batch).unsqueeze(1)
                U = self.user_emb(user_ids)
                b_i = self.item_bias(item_id_batch)
                b_u = self.user_bias(user_ids).T
                mu = self.global_bias
                result.append((self.fc(I_deep * U).squeeze(2) + b_i + b_u + mu))
        return torch.cat(result, dim=0)


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


    # train interaction matrix
    review_ratings_train = review_train[['user_id', 'gmap_id', 'rating']].copy()
    item_user_df_train = review_ratings_train.pivot_table(index='gmap_id', columns='user_id', values='rating', aggfunc='mean')
    item_user_df_train = item_user_df_train.reindex(index=item_all, columns=user_all).sort_index(ascending=True).sort_index(axis=1, ascending=True)
    item_user_matrix_train = torch.tensor(item_user_df_train.fillna(0).to_numpy()).float()
    mask_train = torch.tensor(item_user_df_train.notnull().to_numpy())


    # test interaction matrix
    review_ratings_test = review_test[['user_id', 'gmap_id', 'rating']].copy()
    item_user_df_test = review_ratings_test.pivot_table(index='gmap_id', columns='user_id', values='rating', aggfunc='mean')
    item_user_df_test = item_user_df_test.reindex(index=item_all, columns=user_all).sort_index(ascending=True).sort_index(axis=1, ascending=True)
    item_user_matrix_test = torch.tensor(item_user_df_test.fillna(0).to_numpy()).float()
    mask_test = torch.tensor(item_user_df_test.notnull().to_numpy())


    # dataset
    print('Loading data...')
    embedded_dataset = torch.load('dataset/embedded_dataset.pt')


    # training
    print('Training...')
    batch_size = 1
    embedding_size = 768
    learning_rate = 1e-4
    epochs = 30
    weight_decay = 1e-3
    global_bias = torch.mean(item_user_matrix[mask_train]).item()
    mseloss = MSELoss()

    wandb.init(project="final_proj", name="run")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("GPU Name:", torch.cuda.get_device_name(0))
    lfm_model = LFM(n_item, n_user, 768, global_bias)
    lfm_model.load_state_dict(torch.load('models/lfm_model.pt'))
    item_embedding = lfm_model.item_emb.weight.data
    user_embedding = lfm_model.user_emb.weight.data
    sll_model = SemanticLatentLite(n_item, n_user, 768, 8, global_bias, item_embedding, user_embedding).to(device)
    optimizer = optim.AdamW(sll_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mask_train = mask_train.to(device)
    embedded_dataloader = DataLoader(embedded_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    sll_model.train()

    for epoch in trange(epochs, desc="Epoch"):
        loss_sum = 0
        for step, data in tqdm(enumerate(embedded_dataloader), total=len(embedded_dataloader)):
            item_id, user_ids, desc_inputs, comment_inputs, ratings = data
            item_id = torch.tensor([item_id]).to(device)
            user_ids = user_ids.to(device)
            desc_inputs = desc_inputs.to(device)
            comment_inputs = comment_inputs.to(device)
            ratings = ratings.unsqueeze(0).to(device)
            y_pred = sll_model(item_id, user_ids, desc_inputs, comment_inputs)
            loss = mseloss(y_pred, ratings)
            sll_model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if step % 100 == 99:
                wandb.log({"loss": loss_sum}, step=step+epoch*len(embedded_dataloader))
                loss_sum = 0

    sll_model.eval()
    sll_model.storage()
    torch.save(sll_model.state_dict(), 'models/sll_model.pt')
    print('Model saved.')