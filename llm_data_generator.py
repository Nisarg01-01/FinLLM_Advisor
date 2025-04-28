# ===============================
# Fine-tuning Dataset Generator (Ground Truth Next 10 Days with News Matching)
# ===============================

import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from Informer_structure_patched import load_data

# -------------------------------
# CONFIGURATION
# -------------------------------
INPUT_SEQ_LEN = 60

PRED_LEN = 10
STEP_SIZE = 10

# -------------------------------
# Load News Data from Kaggle File
# -------------------------------
news_df = pd.read_csv("2009_2020.csv")
news_df["date"] = pd.to_datetime(news_df["date"], errors='coerce', utc=True).dt.date
news_df = news_df.dropna(subset=["date"])

# -------------------------------
# Fetch News from Dataset Matching Ticker and Dates
# -------------------------------
def fetch_news_from_dataset(ticker, forecast_dates, news_df):
    headlines = []
    forecast_dt = [datetime.strptime(d, "%Y-%m-%d").date() for d in forecast_dates]
    forecast_min = min(forecast_dt) - timedelta(days=5)
    forecast_max = max(forecast_dt) + timedelta(days=5)

    matching_news = news_df[
        (news_df["stock"] == ticker) &
        (news_df["date"] >= forecast_min) &
        (news_df["date"] <= forecast_max)
    ]

    for _, row in matching_news.iterrows():
        headlines.append(row["title"])
        if len(headlines) >= 3:
            break

    return headlines

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_full_data(df):
    label_encoder = LabelEncoder()
    df["ticker_id"] = label_encoder.fit_transform(df["ticker"])

    features = ["open", "high", "low", "close", "volume"]
    scalers = {col: MinMaxScaler() for col in features}
    for col in features:
        df[col] = scalers[col].fit_transform(df[[col]])

    return df, scalers, label_encoder

# -------------------------------
# Load Training Data
# -------------------------------
df = load_data("Finetune_multi_stock.csv")
df, scalers, label_encoder = preprocess_full_data(df)

# -------------------------------
# Dataset Generation
# -------------------------------
dataset_records = []
tickers = df["ticker"].unique()

for ticker in tickers:
    sub_df = df[df["ticker"] == ticker].reset_index(drop=True)
    if len(sub_df) < INPUT_SEQ_LEN + PRED_LEN:
        continue

    ticker_id = sub_df["ticker_id"].iloc[0]
    dates = sub_df["Date"].astype(str).tolist()

    features = sub_df[["open", "high", "low", "close", "volume"]].values
    ticker_arr = np.full((len(sub_df), 1), ticker_id)
    data_combined = np.hstack([features, ticker_arr])

    for start in range(0, len(sub_df) - INPUT_SEQ_LEN - PRED_LEN + 1, PRED_LEN):  # non-overlapping
        x_window = data_combined[start : start + INPUT_SEQ_LEN]
        y_window = sub_df["close"].iloc[start + INPUT_SEQ_LEN : start + INPUT_SEQ_LEN + PRED_LEN].tolist()
        forecast_dates = dates[start + INPUT_SEQ_LEN : start + INPUT_SEQ_LEN + PRED_LEN]

        scaler = scalers["close"]
        input_scaled = scaler.inverse_transform(x_window[:, 3].reshape(-1, 1)).flatten().tolist()

        # Use scaled y_window directly for calculations
        diff_total = abs(y_window[-1] - y_window[0])  # scaled difference
        step_diffs = np.diff(y_window)                # scaled step differences

        trend = "up" if y_window[-1] > y_window[0] else "down"
        consistent_steps = np.sum(step_diffs > 0) if trend == "up" else np.sum(step_diffs < 0)
        consistency_ratio = consistent_steps / (len(step_diffs) + 1)
        intensity_score = round(float(diff_total * consistency_ratio), 4)

        # AFTER calculation, inverse scale back for storage and output
        output_scaled = scalers['close'].inverse_transform(np.array(y_window).reshape(-1, 1)).flatten().tolist()

        news_list = fetch_news_from_dataset(ticker, forecast_dates, news_df)

        if not news_list:
            continue
        else:
            print(f"News found for {ticker} on {forecast_dates[0]}")

        news_text = "\n".join([f"- {headline}" for headline in news_list])

        input_text = (
            f"Ticker: {ticker}\n"
            f"Trend: {trend} (Intensity Score: {intensity_score})\n"
            f"Forecasted Prices: {output_scaled}\n"
            f"Recent News:\n{news_text}"
        )

        record = {
            "instruction": "Based on the forecast and recent news, provide a trading recommendation.",
            "input": input_text,
            "output": ""  # Leave blank for manual writing
        }
        dataset_records.append(record)

# -------------------------------
# Save Dataset as JSONL
# -------------------------------
with open("finetune_data_kaggle_groundtruth.jsonl", "w") as f:
    for rec in dataset_records:
        f.write(json.dumps(rec) + "\n")

print("✅ Dataset generated and saved as 'finetune_data_kaggle_groundtruth.jsonl'.")