import torch 
from torch.utils.data import Dataset
import torch.nn as nn

class ImplicitDataset(Dataset):
    def __init__(self, train_data):
        self.data = []
        for u, pos, negs in train_data:
            self.data.append((u, pos, 1))  # positive
            for n in negs:
                self.data.append((u, n, 0))  # negative

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, label = self.data[idx]
        return torch.LongTensor([u]), torch.LongTensor([i]), torch.FloatTensor([label])
    
class NCF_MLP(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # đầu ra xác suất
        )

    def forward(self, u, i):
        u_emb = self.user_emb(u).squeeze(1)  # (batch, emb_dim)
        i_emb = self.item_emb(i).squeeze(1)  # (batch, emb_dim)
        x = torch.cat([u_emb, i_emb], dim=1)  # (batch, 2 * emb_dim)
        return self.mlp(x).squeeze(1)  # (batch,)