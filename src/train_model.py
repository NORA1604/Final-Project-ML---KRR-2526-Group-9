import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from data_generator import load_data

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)

def embed(texts):
    with torch.no_grad():
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output = encoder(**encoded)
        return output.last_hidden_state.mean(dim=1)

df = load_data()
X = embed(df["Item"].astype(str).tolist())

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X.numpy())

df["cluster"] = clusters
high_cluster = df.groupby("cluster")["Total_Usage"].mean().idxmax()
df["usage_level"] = df["cluster"].apply(
    lambda x: "HIGH" if x == high_cluster else "LOW"
)

torch.save({
    "kmeans": kmeans,
    "high_cluster": high_cluster
}, "models/usage_classifier.pt")
