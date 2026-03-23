import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GATv2Conv,SAGEConv
import numpy as np

def contrastive_loss(z_graph, z_esm, logit_scale):
    
    # 2. Calculate the similarity matrix [Batch, Batch]
    # Each cell (i, j) is the cosine similarity between graph_i and esm_j
    logits = torch.matmul(z_graph, z_esm.T) * logit_scale
    
    # 3. Create labels: The diagonal elements are the matches
    # labels = [0, 1, 2, ..., BatchSize-1]
    batch_size = z_graph.shape[0]
    labels = torch.arange(batch_size).to(z_graph.device)
    
    # 4. Calculate Cross Entropy in both directions
    # "Which ESM matches this Graph?"
    loss_g2e = F.cross_entropy(logits, labels)
    # "Which Graph matches this ESM?"
    loss_e2g = F.cross_entropy(logits.T, labels)
    
    return (loss_g2e + loss_e2g) / 2

class MultiScaleBlock(nn.Module):
    def __init__(self, esm_dim=1280, hidden_dim=256,dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # 1. LOCAL SCALE: GAT (With Attention)
        # Learns high-resolution, specific neighbor interactions
        self.local_gat = GATv2Conv(esm_dim, hidden_dim, heads=4, concat=True, edge_dim=1,dropout = self.dropout)
        
        # 2. GLOBAL SCALE: SAGE (With Mean/Sum Aggregation)
        # Takes the GAT output and looks 2-hops away to find global context
        self.global_sage = SAGEConv(hidden_dim * 4, esm_dim)
        self.norm = nn.LayerNorm(esm_dim)
        
    def forward(self, x, edge_index, edge_weights, return_attn = False):
        identity = x
        # edge_weights must be [Num_Edges, 1] for GAT
        if edge_weights.dim() == 1:
            edge_weights = edge_weights.unsqueeze(-1)
        # --- STEP 1: LOCAL ---
        # Protein learns about its direct partners using Attention
        if return_attn:
            x, (attn_edge_index, alpha) = self.local_gat(x, edge_index, edge_attr=edge_weights,
                                                         return_attention_weights=return_attn)
        else:
            x, _ = self.local_gat(x, edge_index, edge_attr=edge_weights,
                                                         return_attention_weights=return_attn)
        x = F.elu(x)

        # --- STEP 2: NEIGHBORHOOOD ---
        # Protein looks at the "local summaries" of its neighbors
        # This effectively captures the 2-hop neighborhood
        x = self.global_sage(x, edge_index)
        x = F.relu(x)
        
        x = self.norm(x + identity)

        if return_attn:
            return x, attn_edge_index, alpha
        return x
    
class DeepMultiScalePPI(nn.Module):
    def __init__(self, esm_dim=1280, hidden_dim=256, num_scales=2,proj_dim=128,dropout=0.2
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        current_dim = esm_dim
        for _ in range(num_scales):
            # Each "Scale Block" contains its own GAT + SAGE
            self.layers.append(MultiScaleBlock(current_dim, hidden_dim,dropout))
            current_dim = hidden_dim # Output of SAGE is hidden_dim

        self.graph_proj = nn.Linear(hidden_dim, proj_dim)
        self.esm_proj = nn.Linear(esm_dim, proj_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x, edge_index, edge_weights):
        identity = x
        for layer in self.layers:
            # Pass the informed embedding back into the next neighborhood expansion
            x = layer(x, edge_index, edge_weights)
        
        z_graph = F.normalize(self.graph_proj(x), dim=-1)
        z_esm = F.normalize(self.esm_proj(identity), dim=-1)
        cl_loss = contrastive_loss(z_graph, z_esm, self.logit_scale)
        return cl_loss
    
    @torch.no_grad
    def get_embeddings(self, x, edge_index=None,edge_weights= None):
        if edge_index is not None:
            for layer in self.layers:
                x = layer(x, edge_index, edge_weights)
            z_graph = F.normalize(self.graph_proj(x), dim=-1)
            return z_graph
        else:
            z_esm = F.normalize(self.esm_proj(x), dim=-1)
            return z_esm
    
class DeepScalePrediction(nn.Module):
    def __init__(self, proj_dim=128,dropout = 0.2):
        super().__init__()
        
        self.proj_dim = proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.proj_dim*2, self.proj_dim ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, self.proj_dim // 2 ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim// 2, 1),
            nn.Softplus()  # Output probability
        )
        
    def cat_emb(self, x_emb1,x_emb2):
        return torch.cat([x_emb1 + x_emb2, x_emb1 * x_emb2], dim = -1)

    def forward(self, x_emb1,x_emb2, edge_weights = None):
        combined_emb = self.cat_emb(x_emb1,x_emb2)
        out = self.classifier(combined_emb)
        if edge_weights is not None:
            loss = F.l1_loss(out,edge_weights.unsqueeze(-1))
            return loss, out
        else:
            return out
        
        
        