# nasa_fetch.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 讀取 .env 的 NASA API key
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")

# -----------------------------
# 參數設定
# -----------------------------
lat = 25.0330  # 台北市
lon = 121.5654
days_to_fetch = 30  # 可抓取天數
end_date = datetime.today()
start_date = end_date - timedelta(days=days_to_fetch)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
parameters_str = "T2M,T2M_MAX,T2M_MIN,WS2M,WS10M,RH2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,PS"

params = {
    "start": start_date.strftime("%Y%m%d"),
    "end": end_date.strftime("%Y%m%d"),
    "latitude": lat,
    "longitude": lon,
    "community": "AG",
    "parameters": parameters_str,
    "format": "JSON",
    "user": "kyle"
}

headers = {"Authorization": f"Bearer {NASA_API_KEY}"}

# -----------------------------
# 抓取資料
# -----------------------------
response = requests.get(BASE_URL, params=params, headers=headers)

if response.status_code == 200:
    data = response.json()
    records = []
    parameters = data["properties"]["parameter"]

    dates = list(next(iter(parameters.values())).keys())
    for date in dates:
        records.append({
            "date": date,
            "T2M": parameters["T2M"][date],
            "T2M_MAX": parameters["T2M_MAX"][date],
            "T2M_MIN": parameters["T2M_MIN"][date],
            "WS2M": parameters["WS2M"][date],
            "WS10M": parameters["WS10M"][date],
            "RH2M": parameters["RH2M"][date],
            "ALLSKY_SFC_SW_DWN": parameters["ALLSKY_SFC_SW_DWN"][date],
            "ALLSKY_SFC_LW_DWN": parameters["ALLSKY_SFC_LW_DWN"][date],
            "PS": parameters["PS"][date],
        })

    df = pd.DataFrame(records)

    # 模擬 IoT 感測器數據
    np.random.seed(42)
    df["IoT_Temp"] = df["T2M"] + np.random.normal(0, 0.5, len(df))
    df["IoT_Humidity"] = df["RH2M"] + np.random.normal(0, 2, len(df))

    df.to_csv("heatwave_data.csv", index=False)
    print("✅ NASA POWER data saved to heatwave_data.csv")
else:
    print("❌ Error:", response.status_code, response.text)
