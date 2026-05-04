import pandas as pd
import plotly.express as px

def plot_revenue_trend(df):
    fig = px.line(
        df,
        x="OrderDate",
        y="Revenue",
        markers=True,
        title="Doanh thu theo thời gian"
    )
    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Doanh thu (VND)"
    )
    return fig

def plot_revenue_by_month(df):
    fig = px.bar(
        df,
        x="YearMonth",
        y="Revenue",
        title="Doanh thu theo tháng",
        text="Revenue"
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )
    fig.update_layout(
        xaxis_title="Tháng",
        yaxis_title="Doanh thu (VND)",
        xaxis_type="category"
    )
    return fig

def plot_top_products(df):
    fig = px.bar(
        df,
        x="Product",
        y="Quantity",
        title="Top sản phẩm bán chạy",
        text="Quantity"
    )
    fig.update_layout(
        xaxis_title="Sản phẩm",
        yaxis_title="Số lượng bán"
    )
    return fig

def plot_region_revenue(df):
    fig = px.pie(
        df,
        names="Region",
        values="Revenue",
        title="Doanh thu theo khu vực",
        hole=0.3
    )
    return fig

def plot_cluster_distribution(df):
    all_groups = ["Khách phổ thông", "Khách tiềm năng", "Khách VIP"]

    cluster_count = df["ClusterName"].value_counts().to_dict()

    chart_df = pd.DataFrame({
        "ClusterName": all_groups,
        "Count": [cluster_count.get(group, 0) for group in all_groups]
    })

    fig = px.bar(
        chart_df,
        x="ClusterName",
        y="Count",
        title="Phân bố cụm khách hàng",
        text="Count"
    )
    fig.update_layout(
        xaxis_title="Nhóm khách hàng",
        yaxis_title="Số khách hàng"
    )
    return fig

def plot_forecast(df):
    fig = px.line(
        df,
        x="OrderDate",
        y=["Revenue", "PredictedRevenue"],
        markers=True,
        title="Dự đoán doanh thu"
    )
    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Doanh thu (VND)"
    )
    return fig

def plot_top_customers(df):
    fig = px.bar(
        df,
        x="CustomerName",
        y="Revenue",
        title="Top khách hàng chi tiêu cao",
        text="Revenue"
    )
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )
    fig.update_layout(
        xaxis_title="Khách hàng",
        yaxis_title="Doanh thu (VND)"
    )
    return fig