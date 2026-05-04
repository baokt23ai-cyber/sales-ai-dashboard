import pandas as pd

def revenue_by_date(df):
    result = df.groupby(df["OrderDate"].dt.date)["Revenue"].sum().reset_index()
    result.columns = ["OrderDate", "Revenue"]
    result["OrderDate"] = pd.to_datetime(result["OrderDate"])
    return result

def revenue_by_month(df):
    monthly_df = df.copy()
    monthly_df["YearMonth"] = monthly_df["OrderDate"].dt.strftime("%Y-%m")

    result = (
        monthly_df.groupby("YearMonth", as_index=False)["Revenue"]
        .sum()
        .sort_values("YearMonth")
    )
    return result

def top_products(df, top_n=10):
    result = (
        df.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return result

def revenue_by_region(df):
    result = (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    return result

def top_customers(df, top_n=10):
    result = (
        df.groupby(["CustomerID", "CustomerName"])["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return result

def average_revenue_per_day(df):
    daily_df = revenue_by_date(df)
    if daily_df.empty:
        return 0
    return daily_df["Revenue"].mean()