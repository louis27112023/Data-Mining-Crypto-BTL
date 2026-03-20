# TỔNG HỢP CÁC ĐOẠN CODE "ĂN ĐIỂM" CHÈN VÀO BÁO CÁO (CHƯƠNG 3)

*(Gợi ý: Anh/chị Copy các đoạn Code ngắn bên dưới, đổi Font chữ trong Word thành `Consolas` hoặc bôi nền xám (Shading) rồi dán rải rác vào Chương 3 để Giảng viên thấy mình hiểu sâu về Thuật toán).*

---

### 1. Code Tiền Xử Lý Dữ Liệu Thiếu (Missing Values)

*(Dán vào phần 3.2. DataCleaner)*

```python
def clean_data(df):
    """
    Kỹ thuật điền khuyết Missing Values bằng phương pháp Nội suy (Interpolate) 
    kết hợp fill thuận nghịch đằng sau (bfill/ffill) cho Dữ liệu Chuỗi Thời gian.
    """
    clean_df = df.copy()
    clean_df.replace(' ', pd.NA, inplace=True)
    numeric_cols = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
  
    # Loại bỏ nhiễu và Ép kiểu Float
    for col in numeric_cols:
        clean_df[col] = clean_df[col].astype(str).str.replace(',', '').astype(float)
        clean_df[col] = clean_df[col].interpolate(method='linear')
        clean_df[col] = clean_df[col].bfill().ffill()
      
    return clean_df
```

---

### 2. Code Kỹ thuật Kéo Trễ Dữ liệu (Lagging Features)

*(Dán vào phần 3.2. FeatureBuilder)*

```python
def build_features(df, config):
    """
    Sinh Đặc trưng trễ (Lag) để mô hình có khứu giác về giá quá khứ.
    Đồng thời tính toán Volatility_7d (Độ biến động 7 ngày) làm thang đo rủi ro.
    """
    featured_df = df.copy()
    # Tính lợi nhuận đóng cửa hàng ngày
    featured_df['Close_Return'] = featured_df.groupby('Symbol')['Close'].pct_change()
  
    # Kéo trễ giá trị về quá khứ (T-1, T-3, T-7 ngày)
    for lag in [1, 3, 7]:
        featured_df[f'Close_Lag_{lag}'] = featured_df.groupby('Symbol')['Close'].shift(lag)
      
    # Đo lường Biên độ Rủi ro (Độ lệch chuẩn 7 ngày)
    featured_df['Volatility_7d'] = featured_df.groupby('Symbol')['Close_Return'].rolling(window=7).std().reset_index(0, drop=True)
    return featured_df
```

---

### 3. Code Khai Phá Luật Kết Hợp (Association Rules)

*(Dán vào phần 3.2. Miner - Thuật toán Apriori/FP-Growth)*

```python
from mlxtend.frequent_patterns import apriori, association_rules

def get_rules(b_sets, regime_name, min_support=0.1, min_confidence=0.5):
    """
    Quét thuật toán Apriori qua Ma trận Giỏ hàng (Basket) để 
    khai phá các mẫu hình đồng pha của Crypto trong từng chu kỳ.
    """
    # 1. Tìm tập phổ biến
    frequent_itemsets = apriori(b_sets, min_support=min_support, use_colnames=True)
  
    # 2. Sinh Luật và Bắt lặp Lift > 1.0 (Tránh trùng hợp ngẫu nhiên)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        rules['Vol_Regime'] = regime_name
      
    return rules
```

---

### 4. Code Phân Lớp Random Forest & Bóc Tách Đặc Trưng

*(Dán vào phần 3.2. Trainer - Phân lớp)*

```python
from sklearn.ensemble import RandomForestClassifier

# Khởi tạo Rừng Cây Quyết Định chống Overfitting với class_weight Cân bằng
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Đánh giá mức độ đóng góp (Importances) của từng Feature vĩ mô
importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
print("Top Các yếu tố chi phối mạnh nhất đến xu hướng giá:", importances)
```
