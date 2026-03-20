import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

st.set_page_config(page_title="Crypto AI Analytics", layout="wide", page_icon="💹")

# Add project path to resolve local imports 
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.data.loader import load_config, load_raw_data
from src.data.cleaner import clean_data
from src.features.builder import build_features
from src.mining.association import run_association_mining
from src.mining.clustering import run_clustering
from src.models.supervised import run_classification
from src.models.forecasting import run_forecasting

try:
    config = load_config()
except Exception as e:
    st.error(f"Lỗi đọc cấu hình: {e}")
    st.stop()

# --- HEADER & SIDEBAR ---
with st.sidebar:
    st.title("⚡ Crypto AI Analytics")
    st.markdown("---")
    st.markdown("👉 **Chuyên trang phân tích On-chain & On-data Crypto bằng công nghệ Machine Learning.**")
    st.markdown("👉 **Engine:** Random Forest, Apriori, K-Means, Auto-ARIMA.")
    st.markdown("---")
    st.info("💡 Hướng dẫn: Chuyển qua lại các Tab ở màn hình chính để duyệt qua 4 module phân tích cốt lõi.")

st.title("🚀 Bảng Điều Khiển Phân Tích Tiền Điện Tử (Crypto Dashboard)")
st.markdown("---")

# Nút Load Data dùng chung cache để đỡ tốn thời gian
@st.cache_data
def load_and_preprocess():
    raw_df = load_raw_data(config)
    cleaned_df = clean_data(raw_df)
    featured_df = build_features(cleaned_df, config)
    return raw_df, cleaned_df, featured_df

try:
    with st.spinner("Đang tải Dòng dữ liệu Lịch sử (Khoảng 5 giây)..."):
        raw_df, cleaned_df, featured_df = load_and_preprocess()
except Exception as e:
    st.error(f"Dữ liệu chưa sẵn sàng hoặc có lỗi: {e}")
    st.stop()

# --- METRIC BOARD ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="📊 Tổng Khối Dữ Liệu", value=f"{len(raw_df):,} Date")
with col2:
    st.metric(label="🪙 Mã Token Theo Dõi", value=f"{len(config.get('data', {}).get('coins', []))}")
with col3:
    st.metric(label="🔗 Logic Thuật Toán", value="4 Models")
with col4:
    st.metric(label="🤖 Trạng Thái Hệ Thống", value="Online", delta="Stable")

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data & Đặc Trưng", 
    "🔗 Luật Kết Hợp", 
    "🎯 Phân Cụm Rủi Ro", 
    "📈 Phân Lớp Máy Học", 
    "🔮 Dự Báo Tương Lai"
])

# 1. Tab Data & Preprocessing
with tab1:
    st.subheader("1. Cơ Sở Dữ Liệu & Rút Trích Đặc Trưng Hạt Nhân")
    st.markdown("Khối dữ liệu gốc được tải về, xử lý Missing Values và làm giàu thêm bằng các chỉ báo biến động (Volatility, Return, RSI...).")
    
    st.dataframe(featured_df[['Symbol', 'Date', 'Close', 'Close_Return', 'Volatility_7d', 'Volume_Ratio', 'Price_Trend', 'Vol_Regime']].head(20), use_container_width=True)


# 2. Tab Association Rules
with tab2:
    st.subheader("2. Phát Hiện Luật Giao Dịch Ngầm (Association Rules)")
    st.markdown("Máy học FP-Growth quyét qua lịch sử để rà tìm những cặp Coin có thiên hướng **cùng Tăng** hoặc **cùng Giảm** trong 1 chu kỳ.")
    
    with st.spinner("Khai phá dữ liệu trên hàng chục ngàn Pattern..."):
        frequent_itemsets, rules = run_association_mining(featured_df, config)
    
    if rules is not None and not rules.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"Phát hiện tổng cộng: **{len(rules):,}** luật")
        
        st.markdown("##### 🔥 TOP 10 Luật Xác Suất Đồng Pha Cao Nhất (Confidence lớn nhất)")
        st.dataframe(rules.sort_values(by="confidence", ascending=False).head(10), use_container_width=True)
        
        st.markdown("##### ⚡ Phân rã Luật dành riêng cho Vùng thị trường Rủi ro cao (Regime = High_Vol)")
        high_vol_rules = rules[rules['Regime'] == 'High_Vol']
        if not high_vol_rules.empty:
            st.dataframe(high_vol_rules.sort_values(by="lift", ascending=False).head(10), use_container_width=True)
        else:
            st.warning("Không có luật hợp lệ trong Regime Biến động mạnh.")
    else:
        st.warning("Không tìm thấy luật kết hợp đủ ngưỡng Threshold.")


# 3. Tab Clustering
with tab3:
    st.subheader("3. Hoạch Định Nhóm Tài Sản Rủi Ro (K-Means Clustering)")
    st.markdown(f"Dòng Token tự động được Máy học phân rã thành **{config['mining']['clustering']['n_clusters']} Cụm (Cluster)** dựa trên chỉ số rủi ro (Vol) và độ lớn lợi nhuận (Return).")
    
    with st.spinner("Đang tính toán ranh giới cụm..."):
        clustered_df, cluster_profiles = run_clustering(featured_df, config)
        
    st.markdown("**Hồ sơ Trung Bình Độ Đo Lường Các Cụm Tài Sản:**")
    st.dataframe(cluster_profiles, use_container_width=True)
    
    st.markdown("**Ma Trận Phân Tán Vector:**")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=clustered_df, x='Volatility_7d', y='Close_Return', hue='Cluster', palette='viridis', alpha=0.9, s=80, ax=ax)
    ax.set_title("Kết quả phân nhóm Rủi ro Crypto bằng Machine Learning", fontweight='bold')
    st.pyplot(fig)


# 4. Tab Classification
with tab4:
    st.subheader("4. Dự Đoán Trend Thị Trường Bằng Random Forest")
    st.markdown("AI sử dụng mạng Rừng Cây Quyết Định (Random Forest) để dự đoán nến Tương lai Tăng hay Giảm dựa trên hệ biến On-data.")
    
    with st.spinner("Training Random Forest Classifier..."):
        rf_model, feature_importances = run_classification(featured_df, config)
        
    st.markdown("**Trọng lượng Tầm quan trọng của các Tính Năng Đầu Vào (Feature Importances):**")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='magma', ax=ax2)
    ax2.set_title("Feature Importances Rate - Yếu tố nào ảnh hưởng đến giá trị nhất?", fontweight='bold')
    st.pyplot(fig2)


# 5. Tab Forecasting
with tab5:
    st.subheader("5. Tầm Nhìn Viễn Cảnh Tương Lai (Time Series - Auto ARIMA)")
    st.markdown("Kế thừa thuật toán Hồi quy Chuỗi Thời gian kinh điển (ARIMA), Test vòng Walk-forward khắt khe ra Sai số để làm mô hình Dự báo Giá cho 14 ngày tới.")
    
    st.info("💡 Ấn nút bên dưới để Khởi động máy học. Module này cần từ 15-30 giây để tính toán đạo hàm chuỗi ARIMA.")
    if st.button("🚀 Bắt đầu Chạy Simulation Hồi Quy Auto-ARIMA", type="primary", use_container_width=True):
        with st.spinner("⏳ Engine Đang train Auto-ARIMA trên tập thử nghiệm. Xin vui lòng không tắt trang..."):
            metrics_df = run_forecasting(featured_df, config)
            st.success("✅ Training Thành công! Báo cáo chỉ số Phục dựng:")
            st.dataframe(metrics_df, use_container_width=True)
            st.write("📷 Toàn bộ Kết xuất hình ảnh Đồ thị Đo lường Biến động thực nghiệm (Test/Predict và Residuals) đã tự động đổ vào kho lưu trữ `outputs/figures/`.")

st.markdown("---")
st.caption("2025 © Xây dựng tích hợp bằng Streamlit Framework. Vận hành Python Data Mining Engine.")
