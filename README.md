# 多模態 Transformer 都市熱浪預測

## 📝 專案概述
本專案使用 **時序多模態 Transformer** 模型，結合 **NASA POWER 衛星資料** 與 **IoT 微氣候感測器資料**，預測都市熱浪。目標是提供 **精細化、即時的熱浪預警**。

---

## 🌍 背景
- 全球氣候變遷使都市熱浪頻繁發生，影響居民健康與能源消耗。
- 現有氣象站資料無法完整捕捉都市微氣候差異。
- NASA POWER API 提供高解析度衛星氣象資料（溫度、風速、輻射、氣壓等）。
- 樓頂 IoT 感測器能補充局部微氣候資訊（溫度與濕度）。
- 結合衛星與 IoT 資料，可建立 **Transformer 模型**，學習時空特徵，提高熱浪預測精度。

---

## 🎯 研究目標
1. **資料收集與整合**
   - NASA POWER：每日 T2M、T2M_MAX、T2M_MIN、WS2M、WS10M、RH2M、輻射、氣壓
   - IoT 樓頂感測器：溫度、濕度
   - 將資料整合為滑動窗口序列，作為 Transformer 輸入

2. **模型建構**
   - Temporal Transformer 提取時序特徵
   - 多模態輸入：衛星 + IoT
   - 輸出：熱浪分類（T2M > 30°C）或溫度回歸

3. **驗證與分析**
   - 與氣象署資料比對
   - 評估指標：Loss、Accuracy、F1-score、RMSE
   - Attention Map 可視化，分析模型關注特徵

---

## ⚙️ 實作內容

### 1. 資料抓取
- `nasa_fetch.py`：
  - 抓取 NASA POWER 衛星資料
  - 模擬 IoT 感測器溫濕度
  - 輸出 CSV：`heatwave_data.csv`

### 2. 模型訓練
- `multimodal_transformer_all_features.py`：
  - Temporal Transformer + Linear 層 + Dropout
  - 支援 K-Fold Cross Validation
  - Loss 曲線與 Sample Prediction 可視化
  - 支援多特徵序列輸入（衛星 + IoT）

### 3. 功能特性
- 多模態輸入（衛星 + IoT）
- 時序滑動窗口序列
- 可設定超參數：seq_len、hidden_dim、dropout、epochs
- 支援即時推論

---

## 📊 結果可視化
- Loss 曲線：Train vs Test
- Sample Predictions：熱浪發生情況
- Attention Map（選用）：分析模型重點時間步與特徵

---

## 🚀 創新與優勢
- 捕捉氣象站無法觀測的微氣候變化
- 多模態 Transformer 學習時空依賴
- K-Fold Cross Validation 提升小資料泛化能力
- Attention Map 可視化，提高模型解釋性
- 可延伸至 IoT / Raspberry Pi 邊緣部署

---

## 🔧 環境需求
```bash
pip install torch pandas numpy scikit-learn matplotlib python-dotenv requests

