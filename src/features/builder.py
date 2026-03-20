import pandas as pd
import numpy as np

def build_features(df, config):
    """
    Thực hiện Feature Engineering dựa trên config.
    Tạo các features cần thiết: Daily Return, Lag, Volatility, Binning
    """
    featured_df = df.copy()
    
    # Đảm bảo sort theo thời gian và từng Coin
    featured_df = featured_df.sort_values(by=['Symbol', 'Date'])
    
    # 1. Tính Tỷ suất sinh lời hàng ngày (Daily Return) trên giá Close
    featured_df['Close_Return'] = featured_df.groupby('Symbol')['Close'].pct_change()
    
    # 2. Tính Volatility (Độ biến động theo độ lệch chuẩn)
    for window in config['features'].get('rolling_windows', [7]):
        featured_df[f'Volatility_{window}d'] = featured_df.groupby('Symbol')['Close_Return'].rolling(window=window).std().reset_index(0, drop=True)
        
    # 3. Tính Lags (Giá trị quá khứ)
    for lag in config['features'].get('lag_days', [1, 3]):
        featured_df[f'Close_Lag_{lag}'] = featured_df.groupby('Symbol')['Close'].shift(lag)
        
    # 4. Tính Volume Ratio (Volume thay đổi như thế nào)
    featured_df['Volume_Ratio'] = featured_df.groupby('Symbol')['Volume'].pct_change()
    # Khắc phục lỗi Infinity Value do chia cho Volume bằng 0 (Hoặc siêu nhỏ)
    featured_df['Volume_Ratio'] = featured_df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan)
    featured_df['Close_Return'] = featured_df['Close_Return'].replace([np.inf, -np.inf], np.nan)
        
    # 5. Phân nhóm/Rời rạc hóa (Binning) rành cho Association Rule
    # Bin 1: Trạng thái giá hôm nay (Up / Down / Flat)
    conditions = [
        (featured_df['Close_Return'] > 0.01),
        (featured_df['Close_Return'] < -0.01)
    ]
    choices = ['Up', 'Down']
    featured_df['Price_Trend'] = np.select(conditions, choices, default='Flat')
    
    # Bin 2: Volatility (High / Normal / Low)
    # Phân vị theo toàn thị trường
    high_vol_thresh = featured_df['Volatility_7d'].quantile(0.75)
    low_vol_thresh = featured_df['Volatility_7d'].quantile(0.25)
    
    vol_conds = [
        (featured_df['Volatility_7d'] > high_vol_thresh),
        (featured_df['Volatility_7d'] < low_vol_thresh)
    ]
    vol_choices = ['High_Vol', 'Low_Vol']
    featured_df['Vol_Regime'] = np.select(vol_conds, vol_choices, default='Normal_Vol')
    
    # Xoá NA sinh ra do Shift/Rolling
    featured_df = featured_df.dropna()
    
    return featured_df

if __name__ == "__main__":
    print("Test feature builder module")
