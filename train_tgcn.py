import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric_temporal.nn.recurrent import TGCN
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Set environment variables for reproducibility and safety
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, accuracy_score

# 1. Configuration & Seeding
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def load_tgcn_data(data_dir='data'):
    print("Loading interactions...")
    # Load and process interaction data
    # Only using book_interaction.csv
    file_path = os.path.join(data_dir, 'book_interaction.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    df_inter = pd.read_csv(file_path)
    # Clean column names (strip type suffix if present, e.g. user_id:token -> user_id)
    df_inter.columns = [c.split(':')[0] for c in df_inter.columns]
    
    # Ensure timestamp is datetime
    df_inter['timestamp'] = pd.to_datetime(df_inter['timestamp'])

    print("Mapping IDs...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Encode Users and Items
    df_inter['user_idx'] = user_encoder.fit_transform(df_inter['user_id'].astype(str))
    df_inter['item_idx'] = item_encoder.fit_transform(df_inter['item_id'].astype(str))
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    print(f"Total Users: {num_users}, Total Items: {num_items}")

    print("Creating temporal snapshots...")
    df_inter['month'] = df_inter['timestamp'].dt.to_period('M')
    # Sort by month to ensure temporal order
    months = sorted(df_inter['month'].unique())
    
    # 1. Determine Training Split (70%) to build the static graph
    train_len = int(len(months) * 0.7)
    train_months = months[:train_len]
    
    # 2. Get all interactions in the training set
    df_train = df_inter[df_inter['month'].isin(train_months)]
    
    # 3. Build Static Edge Index from Train Set
    train_u_idx = df_train['user_idx'].values
    train_i_idx = df_train['item_idx'].values
    
    train_u_node_idx = torch.tensor(train_u_idx, dtype=torch.long)
    train_i_node_idx = torch.tensor(train_i_idx + num_users, dtype=torch.long)
    
    # Create undirected graph from unique training interactions
    # Note: torch.unique might be needed if multiple interactions exist, but edge_index usually works fine with multis.
    # We'll use the raw list; duplicates increase weight in message passing or are redundant. 
    # For efficiency and cleanliness, let's keep unique edges.
    train_edges_df = df_train[['user_idx', 'item_idx']].drop_duplicates()
    unique_u_idx = torch.tensor(train_edges_df['user_idx'].values, dtype=torch.long)
    unique_i_idx = torch.tensor(train_edges_df['item_idx'].values + num_users, dtype=torch.long)
    
    train_edge_index = torch.stack([
        torch.cat([unique_u_idx, unique_i_idx]), 
        torch.cat([unique_i_idx, unique_u_idx])
    ], dim=0)
    
    print(f"Static Training Graph created with {train_edges_df.shape[0]} edges.")
    
    dataset = []
    
    for m in months:
        snapshot_df = df_inter[df_inter['month'] == m]
        if snapshot_df.empty:
            continue
            
        u_idx_raw = snapshot_df['user_idx'].values
        i_idx_raw = snapshot_df['item_idx'].values
        
        # Node indices for validataion/testing targets
        u_node_idx = torch.tensor(u_idx_raw, dtype=torch.long)
        i_node_idx = torch.tensor(i_idx_raw + num_users, dtype=torch.long)
        
        # Use the STATIC train_edge_index for Graph Structure
        dataset.append({
            'edge_index': train_edge_index,
            'y': torch.ones(len(snapshot_df), dtype=torch.float), # All interactions are positive (likes)
            'target_u': u_node_idx,
            'target_i': i_node_idx
        })
    
    print(f"Loaded {len(dataset)} snapshots.")
    
    # Split into Train, Val, Test (70/10/20)
    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.1)
    
    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len : train_len + val_len]
    test_dataset = dataset[train_len + val_len :]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, num_users, num_items

# 2. Model Definition
class TGCNRecommender(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_dim=64, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        
        # Learnable Node Embeddings
        # Since we don't have external features, we learn the features from scratch.
        # This replaces the static feature matrix X.
        self.node_emb = nn.Embedding(self.num_nodes, embedding_dim)
        
        # Initialize embeddings (optional, Xavier often good)
        nn.init.xavier_uniform_(self.node_emb.weight)
        
        # T-GCN Layer
        # Input dim is embedding_dim, output dim is embedding_dim (or hidden_dim)
        self.tgcn = TGCN(in_channels=embedding_dim, out_channels=embedding_dim)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.lr = lr
        self.h = None # Hidden state for GRU

    def on_train_epoch_start(self):
        self.h = None # Reset hidden state at start of epoch

    def on_test_epoch_start(self):
        self.h = None # Reset hidden state at start of testing

    def forward(self, edge_index, target_u, target_i, h):
        # 1. Get current node embeddings
        # shape: [num_nodes, embedding_dim]
        # We pass usage of all nodes to the TGCN
        x = self.node_emb.weight
        
        # 2. Update Embeddings with T-GCN
        # h_new shape: [num_nodes, embedding_dim]
        h_new = self.tgcn(x, edge_index, None, h)
        
        # 3. Lookup Embeddings for target pairs from the UPDATED states (h_new)
        u_emb = h_new[target_u]
        i_emb = h_new[target_i]
        
        # 4. Predict Rating
        combined = torch.cat([u_emb, i_emb], dim=1)
        out = self.predictor(combined)
        
        return out, h_new

    def training_step(self, batch, batch_idx):
        # Batch is a signle snapshot dictionary (batch_size=1)
        edge_index, y = batch['edge_index'], batch['y']
        target_u, target_i = batch['target_u'], batch['target_i']
        
        # Handle hidden state persistence
        # If we are starting a new sequence (though here we just iterate snapshots linearly),
        # we keep h. If h is detached, gradients don't flow back to previous timesteps' graph structure 
        # (TBPTT), but here we usually just do 1-step or truncated backprop.
        # Given the loop structure in main, we process one snapshot at a time.
        
        # Ensure h is on the same device
        if self.h is None:
             self.h = torch.zeros(self.num_nodes, self.embedding_dim, device=self.device)
        else:
             self.h = self.h.to(self.device).detach() # Detach to implement Truncated BPTT
        
        y_hat, h_new = self.forward(edge_index, target_u, target_i, self.h)
        
        # Update hidden state
        self.h = h_new.detach()
        
        loss = F.mse_loss(y_hat.view(-1), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, "test")

    def _evaluate_step(self, batch, batch_idx, stage):
        # 1. Forward Pass
        edge_index, y = batch['edge_index'], batch['y']
        target_u, target_i = batch['target_u'], batch['target_i']
        
        if self.h is None:
             self.h = torch.zeros(self.num_nodes, self.embedding_dim, device=self.device)
        else:
             self.h = self.h.to(self.device).detach()
        
        y_hat_pos, h_new = self.forward(edge_index, target_u, target_i, self.h)
        
        # 2. Negative Sampling (1:1)
        num_neg = len(target_u)
        neg_i = torch.randint(
            self.hparams.num_users, 
            self.hparams.num_users + self.hparams.num_items, 
            (num_neg,), 
            device=self.device
        )
        
        u_emb = h_new[target_u]
        neg_i_emb = h_new[neg_i]
        
        combined_neg = torch.cat([u_emb, neg_i_emb], dim=1)
        y_hat_neg = self.predictor(combined_neg)
        
        # 3. Metrics
        pos_probs = torch.sigmoid(y_hat_pos.view(-1))
        neg_probs = torch.sigmoid(y_hat_neg.view(-1))
        
        all_probs = torch.cat([pos_probs, neg_probs])
        all_labels = torch.cat([torch.ones_like(pos_probs), torch.zeros_like(neg_probs)])
        
        preds = (all_probs > 0.5).float()
        
        tp = ((preds == 1) & (all_labels == 1)).sum().float()
        fp = ((preds == 1) & (all_labels == 0)).sum().float()
        fn = ((preds == 0) & (all_labels == 1)).sum().float()
        tn = ((preds == 0) & (all_labels == 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        self.h = h_new.detach()
        
        self.log_dict({
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_accuracy": accuracy
        }, prog_bar=True)
        
        return precision

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# 3. Main Execution
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, num_users, num_items = load_tgcn_data()
    
    # Model Init
    embedding_dim = 64
    model = TGCNRecommender(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=lambda x: x[0], shuffle=False)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1
    )

    print("Starting Training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training Complete!")
    
    print("Starting Testing...")
    trainer.test(model, test_loader)
    print("Testing Complete!")
