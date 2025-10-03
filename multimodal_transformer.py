# multimodal_transformer_with_attention.py
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. 讀取 CSV
# -----------------------------
df = pd.read_csv("heatwave_data.csv")

sat_features = ["T2M", "WS2M", "RH2M"]
iot_features = ["IoT_Temp", "IoT_Humidity"]

X_sat = df[sat_features].values
X_iot = df[iot_features].values

# 標籤：T2M > 30°C 視為熱浪
y = (df["T2M"] > 30).astype(int).values

# 標準化
scaler_sat = StandardScaler()
scaler_iot = StandardScaler()
X_sat = scaler_sat.fit_transform(X_sat)
X_iot = scaler_iot.fit_transform(X_iot)

# 轉 Torch Tensor
X_sat = torch.tensor(X_sat, dtype=torch.float32).unsqueeze(1)  # [batch, seq_len=1, features]
X_iot = torch.tensor(X_iot, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

# -----------------------------
# 2. 定義簡化 Multi-modal Transformer
# -----------------------------
class MultiModalTransformer(nn.Module):
    def __init__(self, sat_dim, iot_dim, hidden_dim=32, num_heads=1, num_classes=2):
        super().__init__()
        self.sat_layer = nn.TransformerEncoderLayer(d_model=sat_dim, nhead=num_heads)
        self.iot_layer = nn.TransformerEncoderLayer(d_model=iot_dim, nhead=num_heads)
        self.sat_encoder = nn.TransformerEncoder(self.sat_layer, num_layers=1)
        self.iot_encoder = nn.TransformerEncoder(self.iot_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(sat_dim + iot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_sat, x_iot):
        sat_feat = self.sat_encoder(x_sat).mean(dim=1)
        iot_feat = self.iot_encoder(x_iot).mean(dim=1)
        fused = torch.cat([sat_feat, iot_feat], dim=-1)
        out = self.fc(fused)
        return out, sat_feat, iot_feat

# -----------------------------
# 3. 訓練
# -----------------------------
model = MultiModalTransformer(sat_dim=X_sat.shape[-1], iot_dim=X_iot.shape[-1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    outputs, sat_feat, iot_feat = model(X_sat, X_iot)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        acc = (outputs.argmax(1) == y).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# -----------------------------
# 4. Attention Heatmap（分開畫 SAT / IoT）
# -----------------------------
sat_attn = sat_feat.detach().numpy().mean(axis=0)
iot_attn = iot_feat.detach().numpy().mean(axis=0)

# SAT heatmap
plt.figure(figsize=(6,1))
sns.heatmap([sat_attn], annot=True, cmap="coolwarm", xticklabels=sat_features, yticklabels=["SAT"])
plt.title("SAT Feature Attention")
plt.show()

# IoT heatmap
plt.figure(figsize=(4,1))
sns.heatmap([iot_attn], annot=True, cmap="coolwarm", xticklabels=iot_features, yticklabels=["IoT"])
plt.title("IoT Feature Attention")
plt.show()
