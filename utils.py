import pandas as pd
import numpy as np
from collections import defaultdict
import random
import torch
from torch.utils.data import DataLoader
from engine import ImplicitDataset
import requests
import json 
import logging

def load_data(dataset_path, dataset_name):
    """Load dataset based on dataset name and path"""
    if dataset_name == "ml-1m":
        rating = pd.read_csv(dataset_path, sep='::', header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "100k":
        rating = pd.read_csv(dataset_path, sep=",", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "lastfm-2k":
        rating = pd.read_csv(dataset_path, sep="\t", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    elif dataset_name == "hetrec":
        rating = pd.read_csv(dataset_path, sep="\t", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
        rating = rating.sort_values(by='uid', ascending=True)
    else:
        # Default format - you can adjust this
        rating = pd.read_csv(dataset_path, sep=",", header=None, 
                           names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
    
    # Reindex users and items (same as in your example)
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = range(len(user_id))
    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    
    item_id = rating[['mid']].drop_duplicates()
    item_id['itemId'] = range(len(item_id))
    rating = pd.merge(rating, item_id, on=['mid'], how='left')
    
    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    
    print(f'Range of userId is [{rating.userId.min()}, {rating.userId.max()}]')
    print(f'Range of itemId is [{rating.itemId.min()}, {rating.itemId.max()}]')

    df = rating.sort_values(by=['userId', 'timestamp'])
    
    return df

def build_datasets(df):
    user_pos_items = defaultdict(list)
    for _, row in df.iterrows():
        user_pos_items[row['userId']].append(row['itemId'])

    all_items = set(df['itemId'].unique())

    train_data, val_data, test_data = [], [], []

    for user, items in user_pos_items.items():
        if len(items) < 3:
            continue

        test_item = items[-1]
        val_item = items[-2]
        train_items = items[:-2]

        neg_items = list(all_items - set(items))
        test_negs = random.sample(neg_items, 99)
        val_negs = random.sample(neg_items, 99)

        test_data.append((user, test_item, test_negs))
        val_data.append((user, val_item, val_negs))

        train_data.append((user, train_items))
        # for pos_item in train_items:
            # sampled_negs = random.sample(neg_items, 4)
            # train_data.append((user, pos_item, sampled_negs))

    return train_data, val_data, test_data, all_items, len(user_pos_items)

def resample_train_data(train_raw_data, all_items, num_negatives=4):
    sampled_data = []
    for user, pos_items in train_raw_data:
        pos_set = set(pos_items)
        neg_pool = list(all_items - pos_set)
        for pos_item in pos_items:
            sampled_negs = random.sample(neg_pool, num_negatives)
            sampled_data.append((user, pos_item, sampled_negs))
    return sampled_data

def train(model, train_raw_data, all_items, optimizer, criterion, val_data, test_data, epochs=10, batch_size=256):
    best_val_hist = 0
    best_test_hist = 0
    best_test_ndcg = 0

    for epoch in range(epochs):
        # Sample lại dữ liệu train mỗi epoch
        sampled_train = resample_train_data(train_raw_data, all_items, num_negatives=4)
        train_dataset = ImplicitDataset(sampled_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        for u, i, label in train_loader:
            optimizer.zero_grad()
            pred = model(u, i)
            loss = criterion(pred, label.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        val_hit, val_ndcg = evaluate(model, val_data)
        test_hit, test_ndcg = evaluate(model, test_data)
        print(f"Validation Hit@10: {val_hit:.4f}, NDCG@10: {val_ndcg:.4f}")
        print(f"Test Hit@10: {test_hit:.4f}, NDCG@10: {test_ndcg:.4f}")
        if val_hit > best_val_hist:
            best_val_hist = val_hit
            best_test_hist = test_hit
            best_test_ndcg = test_ndcg

    return best_test_hist, best_test_ndcg

def ndcg_at_k(r, k):
    r = np.asarray(r)[:k]
    if r.sum() == 0:
        return 0.0
    dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    idcg = 1.0
    return dcg / idcg

def evaluate(model, data, k=10):
    model.eval()
    hits, ndcgs = [], []

    for user, pos_item, neg_items in data:
        items = [pos_item] + neg_items
        users = [user] * len(items)

        users = torch.LongTensor(users).unsqueeze(1)
        items = torch.LongTensor(items).unsqueeze(1)

        with torch.no_grad():
            scores = model(users, items).detach().cpu().numpy().flatten()

        rank = np.argsort(-scores)
        ranked_items = [items[i].item() for i in rank]

        rel = [1 if item == pos_item else 0 for item in ranked_items]

        hits.append(1 if pos_item in ranked_items[:k] else 0)
        ndcgs.append(ndcg_at_k(rel, k))

    return np.mean(hits), np.mean(ndcgs)

def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def send_webhook_message(webhook_url, message, username=None):
    data = {"content": message}
    
    if username:
        data["username"] = username
    
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")