import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from Informer_structure_patched import transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
model = transformer(input_dim=6, d_model=256, n__heads=8, num_layers=4, ff_dim=512, pred=10).to(device)
model.load_state_dict(torch.load("C:/Nisarg/Projects/SML/FinLLM_Advisor/New_best_informer_model.pth"))
model.eval()

# ----------------------------
# Load Saved Scalers (Used During Training)
# ----------------------------
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)

# ----------------------------
# Functions
# ----------------------------
def load_ticker_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna().reset_index(drop=True)
    df = df[["Date", "open", "high", "low", "close", "volume", "ticker"]]
    return df

def preprocess_ticker_data(df, scalers):
    for col in scalers:
        df[col] = scalers[col].transform(df[[col]])
    return df

def prepare_input(df):
    last_60 = df.iloc[-70:-10]  # latest 60 days
    input_features = last_60[["open", "high", "low", "close", "volume"]].values
    ticker_id = df["ticker_id"].iloc[0]
    ticker_arr = np.full((60, 1), ticker_id)
    input_combined = np.hstack([input_features, ticker_arr])
    x_input = torch.tensor(input_combined, dtype=torch.float32).unsqueeze(0).to(device)
    return x_input

def get_ground_truth(df):
    true_10 = df["close"].iloc[-10:].values.tolist()  # last 10 days as ground truth
    return true_10

# ----------------------------
# Testing for One Ticker File
# ----------------------------
file_path = "C:/Nisarg/Projects/SML/FinLLM_Advisor/company_wise/2024-25_AAPL_stock.csv"
df = load_ticker_data(file_path)
df["ticker_id"] = 0  # manually set ticker ID since testing single ticker

df = preprocess_ticker_data(df, scalers)
x_input = prepare_input(df)
y_true = get_ground_truth(df)

with torch.no_grad():
    preds = model(x_input)
    preds = preds.cpu().numpy().flatten()

# ----------------------------
# Inverse Scaling (Correct Scaler from Training)
# ----------------------------
preds_unscaled = scalers['close'].inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
y_true_unscaled = scalers['close'].inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()

# ----------------------------
# RMSE Calculation
# ----------------------------
rmse = np.sqrt(np.mean((y_true_unscaled - preds_unscaled) ** 2))
print(f"✅ RMSE for {file_path}: {rmse:.4f}")

# ----------------------------
# Plotting
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), y_true_unscaled, label='Ground Truth', marker='o')
plt.plot(range(1, 11), preds_unscaled, label='Predicted', marker='x')
plt.title(f"{file_path} - Prediction vs Actual (10 Days)\nRMSE: {rmse:.4f}")
plt.xlabel('Forecast Day')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
ticker_name = file_path.split('_')[2].replace('.csv', '')  # Extract ticker from file path
plt.savefig(f"prediction_plot.png")
plt.show()