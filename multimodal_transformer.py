# multimodal_transformer_all_features.py
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# -----------------------------
# 1. 讀取 CSV
# -----------------------------
df = pd.read_csv("heatwave_data.csv")
feature_cols = [c for c in df.columns if c != "date"]
X_raw = df[feature_cols].values
y_raw = (df["T2M"] > 30).astype(int).values  # 熱浪標籤

# -----------------------------
# 2. 序列化
# -----------------------------
def create_sequences(X, y, seq_len=3):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_raw, y_raw, seq_len=3)

# -----------------------------
# 3. 標準化
# -----------------------------
scaler = StandardScaler()
X_flat = X_seq.reshape(-1, X_seq.shape[-1])
X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)

X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y_seq, dtype=torch.long)

# -----------------------------
# 4. Transformer 模型
# -----------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=1, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x).mean(dim=1)
        out = self.fc(feat)
        return out, feat

# -----------------------------
# 5. K-fold cross-validation
# -----------------------------
kf = KFold(n_splits=5, shuffle=False)  # 時序資料不用 shuffle
all_train_loss, all_test_loss = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_t)):
    X_train_t, X_test_t = X_t[train_idx], X_t[test_idx]
    y_train_t, y_test_t = y_t[train_idx], y_t[test_idx]

    model = TransformerModel(input_dim=X_t.shape[-1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_loss_list, test_loss_list = [], []
    epochs = 50

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())

        # test loss
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_loss_list.append(test_loss)

    all_train_loss.append(train_loss_list)
    all_test_loss.append(test_loss_list)

    print(f"✅ Fold {fold+1} done.")

# -----------------------------
# 6. 繪圖 (平均 Loss)
# -----------------------------
train_mean = np.mean(all_train_loss, axis=0)
test_mean = np.mean(all_test_loss, axis=0)

plt.figure(figsize=(8,4))
plt.plot(train_mean, label="Train Loss")
plt.plot(test_mean, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Loss Curve (All Features, K-Fold Avg)")
plt.legend()
plt.show()

# -----------------------------
# 7. 單次推論示例
# -----------------------------
model.eval()
with torch.no_grad():
    sample_outputs, _ = model(X_t)
    predictions = sample_outputs.argmax(1).numpy()

print("Sample Predictions:", predictions)
