import torch
import torch.nn as nn
import numpy as np
import os

class LLMEnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, embedding_path=None, freeze_emb=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 第一層線性層 (Input -> Hidden)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # === LLM Embedding Initialization Logic ===
        if embedding_path and os.path.exists(embedding_path):
            print(f"[Model] Loading LLM embeddings from {embedding_path}...")
            # Shape: [num_features, llm_dim]
            emb_numpy = np.load(embedding_path) 
            
            # Check dimensions
            if emb_numpy.shape[0] != input_dim:
                print(f"[Warning] Embedding rows ({emb_numpy.shape[0]}) != Input dim ({input_dim}). Skipping init.")
            else:
                emb_tensor = torch.FloatTensor(emb_numpy)
                llm_dim = emb_tensor.shape[1]
                
                # 如果 LLM 維度 != Hidden Dim，我們需要投影
                # 這裡使用 SVD (PCA) 進行降維初始化，或者簡單的隨機投影
                if llm_dim != hidden_dim:
                    print(f"[Model] Projecting LLM dim ({llm_dim}) to Hidden dim ({hidden_dim})...")
                    # 簡單方案：初始化一個投影矩陣將 LLM 空間映射到 Hidden 空間
                    # W_init = Emb @ Projection
                    projection = torch.randn(llm_dim, hidden_dim) / np.sqrt(llm_dim)
                    weights_init = torch.matmul(emb_tensor, projection)
                else:
                    weights_init = emb_tensor
                
                # 賦值給第一層權重 (注意：nn.Linear 權重形狀是 [out, in]，所以要轉置)
                with torch.no_grad():
                    self.input_proj.weight.copy_(weights_init.T)
                    
                if freeze_emb:
                    print("[Model] Freezing first layer weights.")
                    self.input_proj.weight.requires_grad = False
        else:
            print("[Model] No embedding path provided or file not found. Using random init.")

        # 剩下的網絡結構保持一致
        self.net = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.input_proj(x)
        logits = self.net(x).squeeze(1)
        return logits