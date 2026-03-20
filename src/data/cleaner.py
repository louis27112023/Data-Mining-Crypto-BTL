import pandas as pd
import numpy as np

def clean_data(df):
    """
    Làm sạch dữ liệu cơ bản cho tập giá Coin:
    1. Kiểm tra và drop duplicate.
    2. Điền missing values (ffill hoặc drop).
    3. Cắt/xóa cột thừa.
    """
    cleaned_df = df.copy()
    
    # 1. Drop duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 2. Xóa các cột không cần thiết if any (e.g. SNo)
    if 'SNo' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=['SNo'])
        
    # 3. Xử lý giá trị null/missing
    # Với time series tài chính, forward fill là ưu tiên tốt nhất
    cleaned_df = cleaned_df.ffill()
    cleaned_df = cleaned_df.dropna() # Drop nếu dòng đầu tiên bị NA
    
    # 4. Ép kiểu chuẩn cho Symbol/Name
    if 'Symbol' in cleaned_df.columns:
        cleaned_df['Symbol'] = cleaned_df['Symbol'].astype(str)
        
    return cleaned_df

if __name__ == "__main__":
    print("Test cleaner module")
