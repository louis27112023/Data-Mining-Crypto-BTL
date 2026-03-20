import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(df, config):
    """
    Phân cụm (Clustering) các đồng coin trong pipeline.
    Ví dụ: Trung bình hoá Return, Volatility theo từng đồng Coin 
    để phân cụm các coin theo mức độ biến động (high-risk, stable, etc.)
    """
    n_clusters = config.get('mining', {}).get('clustering', {}).get('n_clusters', 4)
    features_to_cluster = config.get('mining', {}).get('clustering', {}).get('features_to_cluster', ['Close_Return', 'Volatility_7d', 'Volume_Ratio'])
    
    print(f"Bắt đầu Clustering với n_clusters={n_clusters} trên các features: {features_to_cluster}...")
    
    # Gom nhóm theo Symbol (coin) để lấy giá trị trung bình làm profile
    coin_profiles = df.groupby('Symbol')[features_to_cluster].mean().reset_index()
    
    if len(coin_profiles) < n_clusters:
        print("Cảnh báo: Số lượng coin ít hơn số cụm k-means. Sẽ tự giảm n_clusters.")
        n_clusters = len(coin_profiles)
        
    X = coin_profiles[features_to_cluster]
    
    # Chuẩn hoá dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Chạy K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.get('project', {}).get('seed', 42), n_init=10)
    coin_profiles['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Tính Silhouette Score để đánh giá (nếu n_clusters > 1)
    if n_clusters > 1 and len(X_scaled) > n_clusters:
        sil_score = silhouette_score(X_scaled, coin_profiles['Cluster'])
        print(f"Silhouette Score cho {n_clusters} cụm: {sil_score:.4f}")
        
    # Mapping kết quả về dataset gốc
    df = df.merge(coin_profiles[['Symbol', 'Cluster']], on='Symbol', how='left')
    
    # --- VẼ BIỂU ĐỒ PHÂN CỤM ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    if 'Close_Return' in coin_profiles.columns and 'Volatility_7d' in coin_profiles.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=coin_profiles, 
            x='Close_Return', 
            y='Volatility_7d', 
            hue='Cluster', 
            palette='viridis', 
            s=120,
            alpha=0.8
        )
        # Gắn nhãn tên từng đồng coin
        for i in range(len(coin_profiles)):
            plt.text(
                coin_profiles['Close_Return'].iloc[i], 
                coin_profiles['Volatility_7d'].iloc[i] + (coin_profiles['Volatility_7d'].max() * 0.01), 
                coin_profiles['Symbol'].iloc[i],
                fontsize=10, ha='center'
            )
        plt.title('Phân cụm Rủi ro - Lợi nhuận (Clustering Scatter)', fontsize=14)
        plt.xlabel('Tỷ suất lợi nhuận (Close_Return)')
        plt.ylabel('Độ biến động (Volatility_7d)')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        figures_dir = os.path.join(base_dir, 'outputs', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, 'clustering_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    return df, coin_profiles

if __name__ == "__main__":
    print("Test clustering module")
