import pandas as pd

def preprocess_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()
    df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df.dropna(subset=["OrderDate", "CustomerID", "Product", "Region", "Quantity", "UnitPrice"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    return df