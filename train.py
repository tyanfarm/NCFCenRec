import pandas as pd 
import argparse
import torch 
import torch.nn as nn
import os  
from utils import *
from engine import ImplicitDataset, NCF_MLP
import datetime


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='NCFCenRec')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--l2_regularization', type=float, default=3*1e-4)
parser.add_argument('--dataset', type=str, default='100k')
args = parser.parse_args()

config = vars(args)
if config['dataset'] == 'ml-1m':
    config['num_users'] = 6040
    config['num_items'] = 3706
elif config['dataset'] == '100k':
    config['num_users'] = 943
    config['num_items'] = 1682
elif config['dataset'] == 'lastfm-2k':
    config['num_users'] = 1600
    config['num_items'] = 12454
elif config['dataset'] == 'hetrec':
    config['num_users'] = 2113
    config['num_items'] = 10109
else:
    pass

folders = ["log"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load dataset
dataset_dir = "data/" + config['dataset'] + "/" + "ratings.dat"
df = load_data(dataset_dir, config['dataset'])
train_data, val_data, test_data, all_items, num_users = build_datasets(df)

# Create model
model = NCF_MLP(num_users, len(all_items), 32)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2_regularization'])
criterion = nn.BCELoss()

all_items=set(df['itemId'].unique())
best_hit, best_ndcg = train(model, train_data, all_items, optimizer, criterion, val_data, test_data, epochs=100)

logging.info(f"Best hit@10: {best_hit:.4f}, best ndcg@10: {best_ndcg:.4f}")

# Save configuration and results
message_discord = f"\n**Dataset: {config['dataset']}**\n```method: {config['alias']}, lr: {config['lr']}, l2-reg: {config['l2_regularization']}\nBest_HR: {best_hit:.4f}, Best_NDCG: {best_ndcg:.4f}```\n"

# Save results to discord
WEBHOOK_URL = "https://discord.com/api/webhooks/1376985100020875464/7RKHQ6UFOQ9PGjKYKQ_X7Kxq54_RznPnXx_nuLC7QazG3ppSzpkU5OWHulEq9XnpypT8"
send_webhook_message(WEBHOOK_URL, message_discord, username="Notification Bot")