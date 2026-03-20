import sys
import os

# Thêm đường dẫn thư mục gốc vào PYTHONPATH để nhận diện được tệp src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data.loader import load_config, load_raw_data
from src.data.cleaner import clean_data
from src.features.builder import build_features
from src.mining.association import run_association_mining
from src.mining.clustering import run_clustering
from src.models.forecasting import run_forecasting
from src.models.supervised import run_classification

def run_data_pipeline():
    print("1. Đọc file cấu hình...")
    config = load_config()
    
    print("2. Tải dữ liệu thô (raw data)...")
    try:
        raw_df = load_raw_data(config)
        print(f"Tổng số dòng ban đầu: {len(raw_df)}")
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return
        
    print("3. Làm sạch dữ liệu...")
    cleaned_df = clean_data(raw_df)
    print(f"Tổng số dòng sau khi clean: {len(cleaned_df)}")
    
    print("4. Sinh đặc trưng (Feature Engineering)...")
    featured_df = build_features(cleaned_df, config)
    print(f"Tổng số dòng sau khi tạo features/bỏ NA: {len(featured_df)}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("5. Lưu dữ liệu đã xử lý (processed data)...")
    processed_dir = os.path.join(base_dir, config['paths']['data_processed'])
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed_coins.csv")
    featured_df.to_csv(processed_path, index=False)
    print(f"--> Đã lưu file: {processed_path}")

    print("\n6. Khởi chạy thuật toán Mining Kế Hợp (Association Rules)...")
    outputs_tables_dir = os.path.join(base_dir, config['paths']['outputs_tables'])
    os.makedirs(outputs_tables_dir, exist_ok=True)
    frequent_itemsets, rules = run_association_mining(featured_df, config)
    if rules is not None and not rules.empty:
        rules_path = os.path.join(base_dir, config['paths']['outputs_tables'], "association_rules.csv")
        rules.to_csv(rules_path, index=False)
        print(f"--> Đã lưu {len(rules)} luật kết hợp vào {rules_path}")

    print("\n7. Phân cụm biến động tiền điện tử (Clustering)...")
    clustered_df, cluster_profiles = run_clustering(featured_df, config)
    cluster_profiles_path = os.path.join(base_dir, config['paths']['outputs_tables'], "coin_clusters.csv")
    cluster_profiles.to_csv(cluster_profiles_path, index=False)
    print(f"--> Đã lưu cluster profile vào {cluster_profiles_path}")
    
    print("\n8. Phân lớp xu hướng nâng cao (Classification - RF)...")
    rf_model, feature_importances = run_classification(featured_df, config)
    
    print("\n9. Dự báo chuỗi thời gian ARIMA (Forecasting)...")
    forecast_results = run_forecasting(featured_df, config)
    forecast_path = os.path.join(base_dir, config['paths']['outputs_tables'], "forecasting_metrics.csv")
    forecast_results.to_csv(forecast_path, index=False)
    print(f"--> Đã lưu kết quả RMSE/MAE tại {forecast_path}\nHoàn tất Toàn Bộ Pipeline!")

if __name__ == "__main__":
    run_data_pipeline()
