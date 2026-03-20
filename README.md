# BÀI TẬP LỚN DATA MINING - ĐỀ TÀI 14: DỰ BÁO GIÁ VÀ BIẾN ĐỘNG CRYPTO

Đây là kho lưu trữ mã nguồn (Repository) chính thức phục vụ cho môn học Khai phá Dữ liệu. Hệ thống xây dựng một pipeline hoàn chỉnh từ tiền xử lý, khai phá luật kết hợp (Association Rules), phân cụm rủi ro (Clustering) đến dự báo chuỗi thời gian (ARIMA) & Phân lớp Random Forest trên tập dữ liệu Lịch sử Tiền điện tử.

## 1. Công nghệ & Ngôn ngữ
- **Ngôn ngữ:** Python 3.9+
- **Thư viện chính:** `pandas`, `numpy`, `scikit-learn`, `pmdarima` (Auto-ARIMA), `mlxtend` (Apriori/FP-Growth), `matplotlib`.

## 2. Các Tính Năng Triển Khai Thực Nghiệm
Đề tài cung cấp các nhánh thực nghiệm kỹ thuật khai phá cốt lõi sau:
- **Khai phá Luật Kết Hợp (Association Rule Mining):** Rời rạc hoá biến động giá thành (Up/Down/High_Vol). Thuật toán FP-Growth tìm tổ hợp đồng coin thường biến động cùng pha theo từng chu kỳ thị trường (Regime).
- **Phân cụm Dữ liệu (Clustering):** Gom nhóm (K-Means) các coin dựa trên hồ sơ rủi ro (Volatility), Tỷ suất lợi nhuận (Return) và Thanh khoản.
- **Phân lớp Dữ liệu (Classification):** Sử dụng Random Forest Classifier (Supervised Learning) dự đoán khả năng thị trường Đi Vàng (Trend Up) dựa trên lịch sử nến ngày. Trích xuất Feature Importances.
- **Dự báo Chuỗi Thời Gian (Time Series Forecasting):** Chạy Time Series Split & Walk-forward validation với Auto-ARIMA để dự đoán giá Tương lai (Close Price). Đánh giá bằng MAE/RMSE và khảo sát phần dư (Residuals Plot).

## 3. Cấu trúc Dự Án (Project Repo Pattern)
Dự án được phân rã thành các module để đảm bảo tính Reproducible:

```text
DATA_MINING_PROJECT/
├── README.md                  # Hướng dẫn sử dụng gốc
├── requirements.txt           # File cài đặt thư viện
├── configs/
│   └── params.yaml            # Siêu tham số cấu hình: seed, paths, hyperparams, dataset...
├── data/
│   ├── raw/                   # Nơi chứa dữ liệu gốc tải từ Kaggle (coin_xxx.csv)
│   └── processed/             # Dữ liệu sạch, đã build feature chuẩn bị scale/train
├── src/                       # Thư mục mã nguồn linh hồn chính
│   ├── data/                  # loader.py, cleaner.py
│   ├── features/              # builder.py (Tạo Lag, Return, Volatility Bins, RSI...)
│   ├── mining/                # association.py, clustering.py
│   ├── models/                # forecasting.py, supervised.py
├── scripts/
│   └── run_pipeline.py        # Kịch bản Tự động kết nối tất cả module chạy một lượt
├── outputs/
│   ├── figures/               # Biểu đồ kết xuất tự động (Scatter_plot, Residuals, Forecasts)
│   ├── tables/                # Kết quả Mining và Modeling Dataframe (Metrics, Rules, Profile)
```

## 4. Hướng Dẫn Cài Đặt và Khởi Chạy
Để chạy lại dự án này trên môi trường của người chấm (Giảng viên) hoặc User khác, vui lòng làm đúng theo các bước sau:

### Bước 1: Chuẩn bị Dữ liệu Thô (Data Source)
Do dataset chứa 22~30 đồng coin có dung lượng hàng trăm MB, quy định Repository không lưu trữ thẳng data lên Git. Vui lòng tải dữ liệu từ trang chính thống của Kaggle:
👉 **[Cryptocurrency Historical Prices Dataset - By Sudalairajkumar](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)**
- Sau khi Tải file nén dạng `.zip` về, hãy giải nén toàn bộ các file định dạng: `coin_Bitcoin.csv`, `coin_Ethereum.csv`, ...
- Di chuyển tất cả chúng vào đường dẫn: `DATA_MINING_PROJECT/data/raw/`.

### Bước 2: Tự Động Khởi Chạy Toàn Bộ Hệ Thống (Auto Run)
Để đơn giản hoá tiến trình chạy trên Windows mà không bị lỗi sao chép lệnh Terminal, hệ thống đã cung cấp một tệp thi hành cấp cao:
- Truy cập vào thư mục cha của dự án (Nằm cùng cấp bậc với chữ DATA).
- Tìm tệp **`run_project.bat`**.
- Click đúp hai lần vào tệp Batch đó. Console sẽ tự tải môi trường Python, thực hiện tuần tự 9 quy trình từ Clean Data, Mining đến Modeling.
- *(Nếu bạn dùng Terminal trực tiếp trên VSCode Windows, gõ: `.venv/Scripts/python.exe DATA_MINING_PROJECT/scripts/run_pipeline.py`)*

### Bước 3: Xem Báo Cáo Kết Quả
Khi màn hình đen báo hoàn tất **Done**, vui lòng mở thư mục `outputs/` để thu thập các thành phẩm thực nghiệm:
- `outputs/tables/`: Thu hoạch Bảng Luật Liên Kết (Association Rules), Dataset Clusters và Bảng Báo Cáo Số Liệu Mô hình (MAE/RMSE/F1).
- `outputs/figures/`: Biểu đồ Scatter Plots phân cụm, Biểu đồ Đường dư (Residuals) và Biểu đồ Forecast Walk-forward đẹp mắt dùng để đính kèm vào File Báo Cáo.

---

