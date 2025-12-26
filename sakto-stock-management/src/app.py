from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

csv_path = DATA_DIR / "retail_store_inventory.csv"
df = pd.read_csv(csv_path)

product_stats = df.groupby("Product ID").agg({
    "Units Sold": "mean",
    "Inventory Level": "mean",
    "Demand Forecast": "mean",
    "Units Ordered": "mean",
    "Competitor Pricing": "mean"
}).reset_index()

app = Flask(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)
encoder.eval()

def embed_text(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        output = encoder(**encoded)
        return output.last_hidden_state.mean(dim=1)

class DemandTrendModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(391, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

model = DemandTrendModel()
trend_model_path = MODELS_DIR / "trend_model.pt"
model.load_state_dict(torch.load(trend_model_path, map_location="cpu"))
label_map_path = MODELS_DIR / "label_map.joblib"
label_map = joblib.load(label_map_path)
inv_label_map = {v: k for k, v in label_map.items()}

label_map = {0: "DECREASING", 1: "STABLE", 2: "INCREASING"}

def adjust_forecast(base, period, season, price, discount):
    period_factor = 1.0 if period == "week" else 1.3
    season_factor = {
        "Spring": 1.1,
        "Summer": 1.2,
        "Autumn": 1.0,
        "Winter": 1.15
    }.get(season, 1.0)

    discount_factor = 1 + (discount / 100)

    return base * period_factor * season_factor * discount_factor

def predict(product_id, period, season, price, discount):
    row = product_stats[product_stats["Product ID"] == product_id].iloc[0]

    adjusted_forecast = adjust_forecast(
        row["Demand Forecast"], period, season, price, discount
    )

    text = f"Product {product_id} season {season}"
    text_emb = embed_text(text)

    numeric = torch.tensor([[
        row["Inventory Level"],
        row["Units Sold"],
        row["Units Ordered"],
        adjusted_forecast,
        price,
        discount,
        row["Competitor Pricing"]
    ]], dtype=torch.float32)

    X = torch.cat([text_emb, numeric], dim=1)

    with torch.no_grad():
        pred = model(X).argmax(dim=1).item()

    trend = label_map[pred]

    if trend == "INCREASING":
        quantity = int(adjusted_forecast * 1.2)
    elif trend == "STABLE":
        quantity = int(adjusted_forecast)
    else:
        quantity = int(adjusted_forecast * 0.8)

    return trend, quantity

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        product_id = request.form["product_id"]
        period = request.form["period"]
        season = request.form["season"]
        price = float(request.form["price"])
        discount = float(request.form["discount"])

        trend, qty = predict(product_id, period, season, price, discount)

        result = {
            "product": product_id,
            "trend": trend,
            "quantity": qty
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
