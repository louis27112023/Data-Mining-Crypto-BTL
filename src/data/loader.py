import os
import pandas as pd
import yaml
import glob

def load_config(config_path="configs/params.yaml"):
    """Đọc file cấu hình params.yaml bằng đường dẫn tuyệt đối"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    abs_config_path = os.path.join(base_dir, config_path)
    with open(abs_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    """
    Đọc và merge dữ liệu nhiều đồng coin từ thư mục data/raw
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, config['paths']['data_raw'])
    target_coins = config['data'].get('coins', [])
    
    # Auto-detect all CSV files if configuration says 'all' or empty
    if not target_coins or target_coins == 'all':
        csv_files = glob.glob(os.path.join(raw_dir, "coin_*.csv"))
        target_coins = [os.path.basename(f) for f in csv_files]
        if not target_coins:
            print("Cảnh báo: Không tự động tìm thấy file coin_*.csv nào trong data/raw")
    elif isinstance(target_coins, str):
        target_coins = [target_coins]
        
    df_list = []
    
    for coin_file in target_coins:
        file_path = os.path.join(raw_dir, coin_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_list.append(df)
        else:
            print(f"Cảnh báo: Không tìm thấy file {file_path}")
            
    if not df_list:
        raise FileNotFoundError("Không có dữ liệu gốc nào được tìm thấy. Vui lòng tải dữ liệu về data/raw")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Ép kiểu Datetime cho cột Date
    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.normalize()
    
    return combined_df

if __name__ == "__main__":
    cfg = load_config()
    print("Test loader...", cfg['project']['name'])
