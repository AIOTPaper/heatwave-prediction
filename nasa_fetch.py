# nasa_fetch.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")  # 從 .env 讀取你的 API key

# -----------------------------
# 設定抓取資料的座標與時間
# -----------------------------
lat = 25.0330  # 台北市中心
lon = 121.5654
end_date = datetime.today()
start_date = end_date - timedelta(days=7)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    "start": start_date.strftime("%Y%m%d"),
    "end": end_date.strftime("%Y%m%d"),
    "latitude": lat,
    "longitude": lon,
    "community": "AG",
    "parameters": "T2M,WS2M,RH2M",
    "format": "JSON",
    "user": "kyle"  # <15字元即可
}

headers = {"Authorization": f"Bearer {NASA_API_KEY}"}  # 可選，增加配額

response = requests.get(BASE_URL, params=params, headers=headers)

if response.status_code == 200:
    data = response.json()
    records = []
    parameters = data["properties"]["parameter"]

    # 將每日資料轉成 DataFrame
    dates = list(next(iter(parameters.values())).keys())
    for date in dates:
        records.append({
            "date": date,
            "T2M": parameters["T2M"][date],      # 氣溫
            "WS2M": parameters["WS2M"][date],    # 風速
            "RH2M": parameters["RH2M"][date]     # 相對濕度
        })

    df = pd.DataFrame(records)

    # 模擬樓頂 IoT 感測器數據
    np.random.seed(42)
    df["IoT_Temp"] = df["T2M"] + np.random.normal(0, 0.5, len(df))
    df["IoT_Humidity"] = df["RH2M"] + np.random.normal(0, 2, len(df))

    df.to_csv("heatwave_data.csv", index=False)
    print("✅ NASA POWER data saved to heatwave_data.csv")
else:
    print("❌ Error:", response.status_code, response.text)
