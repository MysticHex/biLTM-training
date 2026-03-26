"""
 model.py - Model Architecture

 Phase 4: Model Building
 - MultiHeadSelfAttention
 - AttnRetrofitModelV2 (Bidirectional LSTM + Multi-Head Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention untuk time-series
    Setiap head dapat capture pattern temporal yang berbeda
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model harus divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections untuk Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        returns: output [batch, seq_len, d_model], attention_weights [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections dan reshape ke [batch, num_heads, seq_len, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc_out(context)

        return output, attention_weights


class AttnRetrofitModelV2(nn.Module):
    """
    Bidirectional LSTM + Multi-Head Attention untuk Energy Forecasting
    """
    def __init__(
        self,
        n_buildings,
        embedding_dim=50,
        input_dim=14,  # Number of features
        hidden_dim=128,
        num_layers=2,
        num_attention_heads=4,
        output_horizon=24,
        dropout=0.3
    ):
        super(AttnRetrofitModelV2, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.bilstm_output_dim = hidden_dim * 2

        # Building ID Embedding
        self.building_embedding = nn.Embedding(n_buildings, embedding_dim)

        # Input projection
        self.input_projection = nn.Linear(input_dim + embedding_dim, hidden_dim)

        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dim if i == 0 else self.bilstm_output_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if i < num_layers - 1 else 0
            ) for i in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.bilstm_output_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Multi-Head Self-Attention
        self.multihead_attn = MultiHeadSelfAttention(
            d_model=self.bilstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        self.attn_layer_norm = nn.LayerNorm(self.bilstm_output_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.bilstm_output_dim, self.bilstm_output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bilstm_output_dim * 2, self.bilstm_output_dim)
        )

        self.ffn_layer_norm = nn.LayerNorm(self.bilstm_output_dim)

        # Output layers
        self.fc1 = nn.Linear(self.bilstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_horizon)

        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights dengan best practices"""
        for name, param in self.named_parameters():
            if 'lstm' in name and 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'embedding' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif 'weight' in name and 'fc' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, sequence, building_id):
        """
        sequence: [batch, seq_len, input_dim]
        building_id: [batch]
        """
        batch_size, seq_len, _ = sequence.shape

        # Building embedding
        building_emb = self.building_embedding(building_id)
        building_emb = building_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate dan project
        x = torch.cat([sequence, building_emb], dim=-1)
        x = self.input_projection(x)
        x = self.activation(x)

        # Bidirectional LSTM dengan residual connections
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(x)
            lstm_out = self.dropout(lstm_out)

            if i == 0:
                x = lstm_out
            else:
                x = ln(x + lstm_out)

        # Multi-Head Self-Attention
        attn_out, attention_weights = self.multihead_attn(x)
        x = self.attn_layer_norm(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_out)

        # Global average pooling
        context = x.mean(dim=1)

        # Output
        out = self.activation(self.fc1(context))
        out = self.dropout(out)
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        out = F.relu(out) + 1e-6

        return out, attention_weights


def create_model(config, device='cuda'):
    """
    Factory function untuk create model
    """
    model = AttnRetrofitModelV2(
        n_buildings=config['n_buildings'],
        embedding_dim=config['embedding_dim'],
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_attention_heads=config['num_attention_heads'],
        output_horizon=config['output_horizon'],
        dropout=config['dropout']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Architecture: BiLSTM({config['hidden_dim']}) + MultiHeadAttn({config['num_attention_heads']} heads)")

    return model


if __name__ == "__main__":
    # Test model creation
    config = {
        'n_buildings': 100,
        'embedding_dim': 50,
        'input_dim': 14,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_attention_heads': 4,
        'output_horizon': 24,
        'dropout': 0.3
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(config, device)

    # Test forward pass
    batch_size = 4
    seq_len = 168
    test_seq = torch.randn(batch_size, seq_len, config['input_dim']).to(device)
    test_bid = torch.randint(0, config['n_buildings'], (batch_size,)).to(device)

    with torch.no_grad():
        pred, attn = model(test_seq, test_bid)

    print(f"\nTest output shape: {pred.shape}")
    print(f"Attention weights shape: {attn.shape}")
    print("✅ Model test passed!")
