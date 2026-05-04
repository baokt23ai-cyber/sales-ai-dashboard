from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def customer_segmentation(df, n_clusters=3):
    customer_df = df.groupby(["CustomerID", "CustomerName"]).agg({
        "Revenue": "sum",
        "Quantity": "sum",
        "OrderDate": "count"
    }).reset_index()

    customer_df.columns = [
        "CustomerID",
        "CustomerName",
        "TotalRevenue",
        "TotalQuantity",
        "TotalOrders"
    ]

    # Giá trị đơn hàng trung bình
    customer_df["AvgOrderValue"] = customer_df["TotalRevenue"] / customer_df["TotalOrders"]

    if len(customer_df) == 0:
        return customer_df

    if len(customer_df) == 1:
        customer_df["Cluster"] = 0
        customer_df["ClusterName"] = "Khách hàng"
        customer_df["BusinessNote"] = "Không đủ dữ liệu để phân cụm."
        customer_df["ClusterPriority"] = 0
        return customer_df

    n_clusters = min(n_clusters, len(customer_df))

    X = customer_df[["TotalRevenue", "TotalQuantity", "TotalOrders"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer_df["Cluster"] = model.fit_predict(X_scaled)

    cluster_summary = (
        customer_df.groupby("Cluster")["TotalRevenue"]
        .mean()
        .sort_values()
    )
    sorted_clusters = cluster_summary.index.tolist()

    cluster_name_map = {}
    cluster_note_map = {}
    cluster_priority_map = {}

    if n_clusters == 3:
        cluster_name_map[sorted_clusters[0]] = "Khách phổ thông"
        cluster_name_map[sorted_clusters[1]] = "Khách tiềm năng"
        cluster_name_map[sorted_clusters[2]] = "Khách VIP"

        cluster_note_map[sorted_clusters[0]] = "Nhóm có mức chi tiêu thấp hơn, phù hợp với chương trình giữ chân và ưu đãi cơ bản."
        cluster_note_map[sorted_clusters[1]] = "Nhóm có tiềm năng tăng trưởng, nên được chăm sóc để tăng tần suất hoặc giá trị mua hàng."
        cluster_note_map[sorted_clusters[2]] = "Nhóm khách hàng giá trị cao, cần ưu tiên chăm sóc và duy trì lâu dài."

        cluster_priority_map[sorted_clusters[0]] = 1
        cluster_priority_map[sorted_clusters[1]] = 2
        cluster_priority_map[sorted_clusters[2]] = 3

    elif n_clusters == 2:
        cluster_name_map[sorted_clusters[0]] = "Khách phổ thông"
        cluster_name_map[sorted_clusters[1]] = "Khách VIP"

        cluster_note_map[sorted_clusters[0]] = "Nhóm có mức chi tiêu thấp hơn trong tập dữ liệu hiện tại."
        cluster_note_map[sorted_clusters[1]] = "Nhóm có mức chi tiêu cao hơn trong tập dữ liệu hiện tại."

        cluster_priority_map[sorted_clusters[0]] = 1
        cluster_priority_map[sorted_clusters[1]] = 3

    else:
        cluster_name_map[sorted_clusters[0]] = "Khách hàng"
        cluster_note_map[sorted_clusters[0]] = "Không đủ dữ liệu để tách thành nhiều nhóm."
        cluster_priority_map[sorted_clusters[0]] = 0

    customer_df["ClusterName"] = customer_df["Cluster"].map(cluster_name_map)
    customer_df["BusinessNote"] = customer_df["Cluster"].map(cluster_note_map)
    customer_df["ClusterPriority"] = customer_df["Cluster"].map(cluster_priority_map)

    return customer_df