import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import math

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print("Checking for null values...", df.isnull().sum())
    df = df.dropna().reset_index(drop=True)
    print("Data after dropping nulls:", df.shape)
    df = df[["Date", "open", "high", "low", "close", "volume", "ticker"]]
    return df

def preprocess_data(df):
    label_encoder = LabelEncoder()
    df["ticker_id"] = label_encoder.fit_transform(df["ticker"])
    
    for ticker, idx in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"Ticker: {ticker}, ID: {idx}")
    
    features = ["open", "high", "low", "close", "volume"]
    scalers = {col: MinMaxScaler() for col in features}

    for col in features:
        df[col] = scalers[col].fit_transform(df[[col]])

    return df, scalers, label_encoder

class StockForecastDataset(Dataset):
    def __init__(self, df, seq=60, pred=10):
        self.X, self.Y = [], []
        tickers = df["ticker"].unique()

        for ticker in tickers:
            sub = df[df["ticker"] == ticker].reset_index(drop=True)

            if len(sub) < (seq + pred):
                print(f"Skipping ticker {ticker} with insufficient data: {len(sub)}")
                continue

            features = sub[["open", "high", "low", "close", "volume"]].values
            ticker_id = sub["ticker_id"].iloc[0]

            for i in range(len(sub) - seq - pred):
                x = features[i:i+seq]
                y = features[i+seq:i+seq+pred, 3]  # 'close' prices

                ticker_arr = np.full((seq, 1), ticker_id)

                self.X.append(np.hstack([x, ticker_arr]))
                self.Y.append(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.Y[index], dtype=torch.float32)
        return x_tensor, y_tensor

# -------------------------------
# Positional Embedding
# -------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -------------------------------
# Informer Encoder Layer
# -------------------------------
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# -------------------------------
# Final Informer Model (Patched)
# -------------------------------
class transformer(nn.Module):
    def __init__(self, input_dim=6, d_model=256, n__heads=8, num_layers=4, ff_dim=512, pred=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEmbedding(d_model)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n__heads, ff_dim, dropout=0.1) for _ in range(num_layers)
        ])
        # Improved Output Head (dense head)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, pred)
        )

    def forward(self, x):
        x = self.embedding(x)                # Input projection
        x = self.pos_encoding(x)             # Add positional encoding
        for layer in self.encoder_layers:    # Pass through Informer Encoder Layers
            x = layer(x)
        x = x[:, -1, :]                      # Use last time step output instead of global average pooling
        x = self.output_layer(x)             # Output forecast
        return x
