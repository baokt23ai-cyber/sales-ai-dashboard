import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.analysis import (
    revenue_by_date,
    revenue_by_month,
    top_products,
    revenue_by_region,
    top_customers,
    average_revenue_per_day
)
from src.clustering import customer_segmentation
from src.forecasting import forecast_revenue
from src.visualization import (
    plot_revenue_trend,
    plot_revenue_by_month,
    plot_top_products,
    plot_region_revenue,
    plot_cluster_distribution,
    plot_forecast,
    plot_top_customers
)

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Sales AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- TÙY CHỈNH PHÔNG CHỮ VÀ GIAO DIỆN (INTER FONT) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, p, div, label {
        font-family: 'Inter', sans-serif !important;
        color: #1E293B;
    }

    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #1e4e79 !important;
    }

    .chart-note {
        font-size: 15px;
        color: #475569;
        background-color: #f8fafc;
        padding: 10px 14px;
        border-left: 4px solid #1f77b4;
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 18px;
        line-height: 1.6;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>
        HỆ THỐNG AI PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU BÁN HÀNG
    </h1>
    <p style='text-align: center; font-size: 18px; color: #555; margin-top: 8px;'>
        Ứng dụng hỗ trợ phân tích doanh thu, hành vi khách hàng và xu hướng kinh doanh từ dữ liệu bán hàng
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- CÁC HÀM HỖ TRỢ ---
def format_list_items(items):
    if not items: return "Không xác định"
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} và {items[1]}"
    return ", ".join(items[:-1]) + f" và {items[-1]}"

def format_currency_short(value):
    if value >= 1_000_000_000: return f"{value / 1_000_000_000:.2f} tỷ VND"
    if value >= 1_000_000: return f"{value / 1_000_000:.2f} triệu VND"
    return f"{value:,.0f} VND"

# --- TẢI FILE (CHO PHÉP TẢI NHIỀU FILE) ---
uploaded_files = st.file_uploader("Tải file dữ liệu bán hàng (.csv)", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        temp_df = load_data(file)
        all_dfs.append(temp_df)
    
    # Kết hợp dữ liệu từ các file đã chọn
    df = pd.concat(all_dfs, ignore_index=True)

    required_columns = ["OrderDate", "CustomerID", "Product", "Region", "Quantity", "UnitPrice"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Thiếu cột bắt buộc: {missing_cols}")
        st.stop()

    df = preprocess_data(df)

    # --- SIDEBAR BỘ LỌC ---
    st.sidebar.title("Bộ lọc dữ liệu")
    regions = sorted(df["Region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Chọn khu vực", options=regions, default=regions)

    if selected_regions:
        df = df[df["Region"].isin(selected_regions)]

    if df.empty:
        st.warning("Không có dữ liệu phù hợp sau khi lọc.")
        st.stop()

    customer_count = df["CustomerID"].nunique()
    n_clusters = min(3, customer_count)
    st.sidebar.markdown(f"**Số cụm khách hàng đang dùng:** {n_clusters}")

    # --- TÍNH TOÁN CHỈ SỐ ---
    total_revenue = df["Revenue"].sum()
    total_units_sold = int(df["Quantity"].sum()) 
    total_customers = df["CustomerID"].nunique()
    total_products = df["Product"].nunique()

    revenue_df = revenue_by_date(df)
    monthly_revenue_df = revenue_by_month(df)
    product_df = top_products(df)
    region_df = revenue_by_region(df)
    top_customer_df = top_customers(df)
    avg_daily_revenue = average_revenue_per_day(df)

    cluster_df = customer_segmentation(df, n_clusters=n_clusters)
    forecast_df, model = forecast_revenue(df)
    forecast_df["PredictedRevenue"] = forecast_df["PredictedRevenue"].round(0).astype(int)

    # --- 1. TỔNG QUAN HỆ THỐNG ---
    st.subheader("1. Tổng quan hệ thống")
    st.markdown('<div class="chart-note">Phần này hiển thị các chỉ số kinh doanh cơ bản được tính từ dữ liệu sau tiền xử lý.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng doanh thu", format_currency_short(total_revenue))
    c2.metric("Tổng sản phẩm đã bán", f"{total_units_sold:,}")
    c3.metric("Tổng khách hàng", total_customers)
    c4.metric("Tổng loại sản phẩm", total_products)
    c5.metric("Doanh thu TB/ngày", f"{avg_daily_revenue:,.0f} VND")

    st.markdown("---")

    # --- 2. DASHBOARD PHÂN TÍCH ---
    st.subheader("2. Dashboard phân tích dữ liệu")
    st.markdown('<div class="chart-note">Các biểu đồ dưới đây giúp quan sát doanh thu, khu vực, sản phẩm bán chạy và khách hàng.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_revenue_trend(revenue_df), use_container_width=True)
        st.markdown('<div class="chart-note">Biểu đồ thể hiện biến động doanh thu thực tế theo từng ngày.</div>', unsafe_allow_html=True)
    with col2:
        st.plotly_chart(plot_revenue_by_month(monthly_revenue_df), use_container_width=True)
        st.markdown('<div class="chart-note">Biểu đồ tổng hợp doanh thu theo từng tháng để quan sát xu hướng.</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_region_revenue(region_df), use_container_width=True)
        st.markdown('<div class="chart-note">Biểu đồ thể hiện tỷ trọng doanh thu giữa các khu vực bán hàng.</div>', unsafe_allow_html=True)
    with col4:
        st.plotly_chart(plot_top_products(product_df), use_container_width=True)
        st.markdown('<div class="chart-note">Biểu đồ cho biết những sản phẩm có số lượng bán ra cao nhất.</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(plot_cluster_distribution(cluster_df), use_container_width=True)
        st.markdown('<div class="chart-note">Phân cụm khách hàng bằng thuật toán K-Means.</div>', unsafe_allow_html=True)
    with col6:
        st.plotly_chart(plot_top_customers(top_customer_df), use_container_width=True)
        st.markdown('<div class="chart-note">Biểu đồ hiển thị các khách hàng có tổng mức chi tiêu cao nhất.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- 3. DỰ ĐOÁN XU HƯỚNG ---
    st.subheader("3. Dự đoán xu hướng doanh thu")
    st.markdown('<div class="chart-note">Sử dụng Linear Regression để ước lượng xu hướng tương lai dựa trên dữ liệu lịch sử.</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
    st.markdown('<div class="chart-note">Đường dự đoán xu hướng doanh thu tổng quát theo thời gian.</div>', unsafe_allow_html=True)

    # --- 4. INSIGHT NHANH ---
    st.markdown("---")
    st.subheader("4. Insight nhanh")
    st.markdown('<div class="chart-note">Các insight được rút ra tự động để hỗ trợ quan sát nhanh.</div>', unsafe_allow_html=True)

    top_region = format_list_items(region_df[region_df["Revenue"] == region_df["Revenue"].max()]["Region"].tolist()) if not region_df.empty else "N/A"
    top_product = format_list_items(product_df[product_df["Quantity"] == product_df["Quantity"].max()]["Product"].tolist()) if not product_df.empty else "N/A"

    ci1, ci2 = st.columns(2)
    with ci1:
        st.success(f"Khu vực có doanh thu cao nhất: **{top_region}**")
    with ci2:
        st.success(f"Sản phẩm bán chạy nhất: **{top_product}**")

    # --- 5. GIẢI THÍCH PHÂN CỤM ---
    st.markdown("---")
    st.subheader("5. Giải thích kết quả phân cụm khách hàng")
    
    cluster_explain = cluster_df.groupby(["ClusterName", "BusinessNote", "ClusterPriority"]).agg(
        SoKhachHang=("CustomerID", "count"),
        DoanhThuTB=("TotalRevenue", "mean")
    ).reset_index().sort_values("ClusterPriority")

    for _, row in cluster_explain.iterrows():
        st.markdown(f"""
            <div style="padding:16px; margin-bottom:12px; border-radius:10px; background-color:#f8fafc; border-left:6px solid #1f77b4;">
                <h4 style="margin:0;">{row['ClusterName']} ({int(row['SoKhachHang'])} khách)</h4>
                <p style="margin:4px 0;"><b>Doanh thu trung bình:</b> {row['DoanhThuTB']:,.0f} VND</p>
                <p style="margin:0;"><b>Chiến lược:</b> {row['BusinessNote']}</p>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("Xem bảng dữ liệu chi tiết"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("Vui lòng tải các tệp CSV để bắt đầu phân tích.")