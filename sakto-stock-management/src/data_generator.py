import pandas as pd

def load_data(path="data/Consumables Report - Oct. 2022.xlsx"):
    df = pd.read_excel(path)

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df.rename(columns={df.columns[0]: "Item"}, inplace=True)

    usage_cols = df.columns[1:]
    df[usage_cols] = df[usage_cols].apply(pd.to_numeric, errors="coerce")

    df["Total_Usage"] = df[usage_cols].sum(axis=1)
    df["Avg_Usage"] = df["Total_Usage"] / len(usage_cols)

    return df[["Item", "Total_Usage", "Avg_Usage"]]
