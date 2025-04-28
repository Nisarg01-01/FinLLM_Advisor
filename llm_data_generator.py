import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from Informer_structure import load_data, transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

INPUT_SEQ_LEN = 60
PRED_LEN = 10
STEP_SIZE = 10

with open("Code/polygon_news_sample.json", "r") as f:
    news_data = json.load(f)


def generate_forecast_dates(last_date_str, forecast_len=PRED_LEN):
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(forecast_len)]
    return forecast_dates

def fetch_news_from_dataset(ticker, forecast_dates, news_data):
    headlines = []

    ticker_to_company_query = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "META": "Meta Platforms",
        "AMZN": "Amazon",
        "NVDA": "Nvidia",
        "TSLA": "Tesla",
        "JPM": "JPMorgan Chase",
        "GS": "Goldman Sachs",
        "BAC": "Bank of America",
        "MS": "Morgan Stanley",
        "PG": "Procter & Gamble",
        "WMT": "Walmart",
        "COST": "Costco",
        "NKE": "Nike",
        "PFE": "Pfizer",
        "JNJ": "Johnson & Johnson",
        "LLY": "Eli Lilly",
        "XOM": "ExxonMobil",
        "CVX": "Chevron"
    }

    company_query = ticker_to_company_query.get(ticker, ticker)

    forecast_dt = [datetime.strptime(d, "%Y-%m-%d") for d in forecast_dates]
    forecast_min = min(forecast_dt) - timedelta(days=3)
    forecast_max = max(forecast_dt) + timedelta(days=3)

    for article in news_data:
        article_tickers = article.get("tickers", [])
        published_utc = article.get("published_utc", "")
        if not published_utc:
            continue

        published_date = datetime.strptime(published_utc.split("T")[0], "%Y-%m-%d")
        title = article.get("title", "").lower()
        description = article.get("description", "").lower()

        ticker_match = ticker in article_tickers
        company_match = company_query.lower() in title or company_query.lower() in description

        if (ticker_match or company_match) and forecast_min <= published_date <= forecast_max:
            headlines.append(article.get("title", ""))
            if len(headlines) >= 3:
                break

    return headlines

def preprocess_full_data(df):
    label_encoder = LabelEncoder()
    df["ticker_id"] = label_encoder.fit_transform(df["ticker"])

    features = ["open", "high", "low", "close", "volume"]
    scalers = {col: MinMaxScaler() for col in features}
    for col in features:
        df[col] = scalers[col].fit_transform(df[[col]])

    return df, scalers, label_encoder


model = transformer(input_dim=6, pred=PRED_LEN, d_model=128, n__heads=4, num_layers=3, ff_dim=256).to(device)
model.load_state_dict(torch.load("Code/best_informer_model.pth"))
model.eval()


df = load_data("Code/multi_stock.csv")
df, scalers, label_encoder = preprocess_full_data(df)

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

    for start in range(0, len(sub_df) - INPUT_SEQ_LEN - PRED_LEN + 1, STEP_SIZE):
        x_window = data_combined[start : start + INPUT_SEQ_LEN]
        x_input = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(device)

        last_input_date = dates[start + INPUT_SEQ_LEN - 1]
        forecast_dates = generate_forecast_dates(last_input_date)

        with torch.no_grad():
            preds = model(x_input)
            forecast = preds[0].cpu().numpy()

        first_pred, last_pred = forecast[0], forecast[-1]
        trend = "up" if last_pred > first_pred else "down"

        scaler = scalers["close"]
        forecasted_prices = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten().tolist()

        first_pred, last_pred = forecasted_prices[0], forecasted_prices[-1]
        trend = "up" if last_pred > first_pred else "down"

        diff_total = abs(last_pred - first_pred)
        step_diffs = np.diff(forecasted_prices)

        if trend == "up":
            consistent_steps = np.sum(step_diffs > 0)
        else:
            consistent_steps = np.sum(step_diffs < 0)

        consistency_ratio = consistent_steps / (len(step_diffs) + 1)
        intensity_score = diff_total * consistency_ratio
        intensity_score = round(float(intensity_score), 4)

        scaler = scalers["close"]
        forecasted_prices = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten().tolist()

        news_list = fetch_news_from_dataset(ticker, forecast_dates, news_data)

        if not news_list:
            print(f"⚠️ Skipping {ticker}: no news found for forecast window.")
            continue

        news_text = "\n".join([f"- {headline}" for headline in news_list])

        input_text = (
            f"Ticker: {ticker}\n"
            f"Trend: {trend} (Intensity Score: {intensity_score})\n"
            f"Forecasted Prices: {forecasted_prices}\n"
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
with open("finetune_data_kaggle.jsonl", "w") as f:
    for rec in dataset_records:
        f.write(json.dumps(rec) + "\n")

print("✅ Dataset generated and saved as 'finetune_data_kaggle.jsonl'.")
