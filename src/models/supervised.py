import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

def run_classification(df, config):
    """
    Phân lớp (Classification) dự đoán biến Price_Trend (Up/Down/Flat)
    X: Các feature quá khứ (Lag, Volatility, Volume Ratio)
    Y: Price_Trend
    """
    print("Bắt đầu Supervised Learning (Phân lớp Classification)...")
    
    # Chuẩn bị dữ liệu
    # Feature X
    features = ['Close_Lag_1', 'Close_Lag_3', 'Volatility_7d', 'Volume_Ratio']
    
    classification_df = df.dropna(subset=features + ['Price_Trend']).copy()
    
    if len(classification_df) == 0:
         print("Dữ liệu trống sau khi dọn dẹp features cho Classifier.")
         return None
         
    X = classification_df[features]
    
    # Target Y (Bỏ nhãn Flat nếu muốn binary classification, hoặc mã hoá)
    le = LabelEncoder()
    y = le.fit_transform(classification_df['Price_Trend'])
    
    # Tách tập train/test (Nên tách theo thời gian thay vì random cho Time Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Huấn luyện mô hình Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    # Đánh giá
    print("========= CLASSIFICATION REPORT =========")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # ROC-AUC (Multiclass OvR)
    try:
         roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
         print(f"ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
         print("Không thể tính ROC-AUC OVR (Có thể do thiếu classes trong test set)")
         
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Feature Importance
    importances = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(importances)
    
    # Save the feature importance plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=importances, x='Importance', y='Feature', palette='magma', ax=ax)
    ax.set_title("Random Forest Feature Importances", fontweight='bold')
    
    out_dir = os.path.join(config.get('paths', {}).get('figures', 'outputs/figures'))
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rf_feature_importance.png'), dpi=300)
    plt.close()
    
    return rf, importances

if __name__ == "__main__":
    print("Test supervised classification module")
