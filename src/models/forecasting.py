import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings("ignore")

def run_forecasting(df, config):
    """
    Dự báo chuỗi thời gian cho giá Đóng cửa (Close)
    Áp dụng auto_arima để tìm tham số tốt nhất (p, d, q)
    Split train/test theo thời gian.
    """
    forecasting_cfg = config.get('modeling', {}).get('forecasting', {})
    train_split = forecasting_cfg.get('train_split', 0.8)
    
    print("Bắt đầu Time Series Forecasting (ARIMA)...")
    
    # Dự báo thử cho 1 đồng coin lớn (VD: Bitcoin) để làm báo cáo mẫu
    results = []
    
    # Lọc danh sách top coin muốn chạy (hoặc chạy tất cả nhưng dễ lâu)
    target_coins = ['AAVE']
    
    for coin in target_coins:
        coin_data = df[df['Symbol'] == coin].copy()
        if coin_data.empty:
            continue
            
        coin_data = coin_data.sort_values('Date').set_index('Date')
        
        # Chọn biến mục tiêu 'Close'
        y = coin_data['Close']
        
        # Split data (Time Series Split)
        train_size = int(len(y) * train_split)
        train, test = y.iloc[:train_size], y.iloc[train_size:]
        
        print(f"[{coin}] Training: {len(train)} days, Testing: {len(test)} days")
        
        # Fit Auto ARIMA
        try:
            model = auto_arima(train, 
                               seasonal=False, 
                               trace=False, 
                               error_action='ignore', 
                               suppress_warnings=True, 
                               stepwise=True)
            
            # Walk-forward prediction
            predictions = []
            print(f"[{coin}] Bắt đầu dự báo walk-forward cho {len(test)} ngày...")
            for i in range(len(test)):
                pred = model.predict(n_periods=1)[0]
                predictions.append(pred)
                model.update(test.iloc[i:i+1])
            
            predictions = pd.Series(predictions, index=test.index)
            
            # Đánh giá (Metrics) tổng thể
            mae = mean_absolute_error(test, predictions)
            rmse = np.sqrt(mean_squared_error(test, predictions))
            smape = np.mean(2.0 * np.abs(test - predictions) / (np.abs(test) + np.abs(predictions))) * 100
            
            # Kết hợp Volatility Regime
            test_extended = coin_data.loc[test.index].copy()
            test_extended['Forecast'] = predictions
            test_extended['Residuals'] = test - predictions
            
            regime_metrics = {}
            for reg in ['High_Vol', 'Low_Vol', 'Normal_Vol']:
                reg_data = test_extended[test_extended['Vol_Regime'] == reg]
                if not reg_data.empty:
                    regime_metrics[f'MAE_{reg}'] = mean_absolute_error(reg_data['Close'], reg_data['Forecast'])
                else:
                    regime_metrics[f'MAE_{reg}'] = None
            
            # --- VẼ BIỂU ĐỒ ---
            import matplotlib.pyplot as plt
            import os
            
            # 1. Biểu đồ Dự báo chính
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train, label='Train Data (Giá đóng cửa)', color='blue')
            plt.plot(test.index, test, label='Thực tế (Test Data)', color='green')
            plt.plot(test.index, predictions, label='Dự báo (Walk-forward ARIMA)', color='red', linestyle='dashed')
            plt.title(f'Dự báo giá {coin} bằng mô hình ARIMA (Walk-forward)', fontsize=14)
            plt.xlabel('Thời gian')
            plt.ylabel('Giá (USD)')
            plt.legend()
            plt.grid(True)
            
            # Lưu hình
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            figures_dir = os.path.join(base_dir, 'outputs', 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            plt.savefig(os.path.join(figures_dir, f'forecast_{coin}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Baseline Naive (Giá ngày mai = Giá hôm nay)
            naive_predictions = test.shift(1)
            naive_predictions.iloc[0] = train.iloc[-1]
            naive_mae = mean_absolute_error(test, naive_predictions)
            naive_rmse = np.sqrt(mean_squared_error(test, naive_predictions))
            
            # 3. Phân tích Phần dư (Residuals)
            residuals = test_extended['Residuals']
            plt.figure(figsize=(12, 4))
            plt.plot(test.index, residuals, color='purple', label='Phần dư (Residuals)')
            plt.axhline(0, color='red', linestyle='dashed')
            plt.title(f'Phân tích Phần dư (Residuals) mô hình ARIMA - {coin}', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, f'residuals_{coin}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            result_dict = {
                'Symbol': coin,
                'Model': str(model),
                'MAE': mae,
                'RMSE': rmse,
                'sMAPE (%)': smape,
                'Naive_MAE': naive_mae,
                'Naive_RMSE': naive_rmse
            }
            result_dict.update(regime_metrics)
            results.append(result_dict)
            
            print(f"[{coin}] Hoàn tất. MAE: {mae:.2f}, RMSE: {rmse:.2f}, sMAPE: {smape:.2f}%")
        except Exception as e:
            print(f"Lỗi khi chạy ARIMA cho {coin}: {e}")
            
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    print("Test forecasting module")
