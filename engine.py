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
    
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()

        # Embeddings for GMF
        self.gmf_user_emb = nn.Embedding(num_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, emb_dim)

        # Embeddings for MLP
        self.mlp_user_emb = nn.Embedding(num_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, emb_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Final prediction layer (combines GMF + MLP output)
        self.output_layer = nn.Linear(emb_dim + 8, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights with Gaussian
        for layer in self.modules():
            if isinstance(layer, (nn.Embedding, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if isinstance(layer, nn.Linear) and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, user, item):
        # GMF path: element-wise product
        gmf_u = self.gmf_user_emb(user).squeeze(1)
        gmf_i = self.gmf_item_emb(item).squeeze(1)
        gmf_out = gmf_u * gmf_i

        # MLP path: concatenate then MLP
        mlp_u = self.mlp_user_emb(user).squeeze(1)
        mlp_i = self.mlp_item_emb(item).squeeze(1)
        mlp_input = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp(mlp_input)

        # Concatenate GMF and MLP outputs and predict
        x = torch.cat([gmf_out, mlp_out], dim=1)
        return self.sigmoid(self.output_layer(x)).squeeze(1)