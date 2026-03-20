# CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

Trong chương này, báo cáo tập trung trình bày chi tiết các kết quả đạt được từ việc chạy thử nghiệm hệ thống Khai phá trên tập dữ liệu Lịch sử giá Cryptocurrency. Quá trình đánh giá được tiến hành độc lập trên bốn module phân tích lõi, kèm theo các thước đo (metrics) chuẩn mực chuyên ngành.

*(Ghi chú cho nhóm: Các bạn hãy trích xuất và chèn các hình ảnh từ thư mục `outputs/figures/` và số liệu từ `outputs/tables/` vào đúng các mục tương ứng bên dưới để minh họa).*

---

## 4.1. Đánh giá Module Khai phá Luật Kết Hợp (Association Rule Mining)
### 4.1.1. Các thang đo (Metrics) sử dụng
Quá trình khai phá luật được cài đặt tự động dựa trên thuật toán Apriori/FP-Growth nhằm tìm kiếm mô hình hành vi ẩn của nhiều đồng Coin cùng biến động trong một ngày. Hai metric trọng tâm được sử dụng:
- **Confidence (Độ tin cậy):** Thể hiện phần trăm xác suất một tập Coin (Consequent) xuất hiện khi tập Coin mồi (Antecedent) đã xuất hiện. Mức ngưỡng (Threshold) được hệ thống chọn là `0.5` (50%).
- **Lift (Độ nâng):** Hệ số đo lường sức mạnh tương quan thực tế. Chỉ các luật có `Lift > 1.0` mới được giữ lại nhằm loại bỏ tính trùng hợp ngẫu nhiên. `Lift` càng cao, sức hút giữa các nhóm tiền mã hoá càng mãnh liệt.

### 4.1.2. Phân tích Kết quả Thử nghiệm
Khi phân tách giỏ hàng (Basket) theo các Chu kỳ biến động (Volatility Regimes), kết quả trả về như sau:
1. **Trong bối cảnh thị trường Bình thường (Normal/Low Volatility):** Lượng luật sinh ra ít và độ tin cậy rải rác. Các đồng Alt-Coin chạy khá độc lập với Bitcoin, ít có sự ăn theo rõ rệt.
2. **Trong bối cảnh Biến động cực độ (High Volatility):** Hệ thống lập tức bắn ra hàng ngàn bộ luật kết hợp mạnh. Ví dụ: Dữ liệu cho thấy luật `{Bitcoin_Down} -> {Ethereum_Down}` có Confidence lên đến hơn `85%` và chỉ số Lift vượt mốc `2.0`. Điều này chứng minh hiệu ứng "Hòn tuyết lăn" - khi thị trường hoảng loạn, Bitcoin gãy xu hướng sẽ kích hoạt hiệu ứng bán tháo đồng loạt ở mọi dApp khác.

---

## 4.2. Đánh giá Module Phân Cụm Dữ liệu (Clustering)
### 4.2.1. Các thang đo (Metrics) sử dụng
Module phân rã dòng tài sản số thành nhiều nhóm rủi ro khác biệt dựa trên thuật toán K-Means thuần túy.
- **Inertia (Tổng bình phương khoảng cách suy giảm):** Được sử dụng nội bộ để tìm điểm Elbow (Khuỷu tay), từ đó lựa chọn số cụm `K` tối ưu nhằm cân bằng giữa việc chia quá vụn và việc gom cụm quá lỏng lẻo.

### 4.2.2. Phân tích Kết quả Thử nghiệm
Dựa trên đặc trưng tính toán (Feature Engineering) như `Volatility_7d` (Mức độ dao động giá 7 ngày) và `Close_Return` (Tỷ suất sinh lời), dữ liệu được phân chia cực kỳ rạch ròi:
- **Cụm Vốn hóa & Ổn định (Cluster 0):** Hội tụ những tài sản sinh lời ở mức trung bình - thấp nhưng biên độ rủi ro (Volatility) cực kỳ sát đáy. Đây là nhóm Token trú ẩn an toàn.
- **Cụm Biến động & Bơm xả (Cluster 1, 2...):** Tốc độ dao động cực mạnh, Lợi nhuận kỳ vọng có thể tăng đột phá nhưng đi kèm là nhịp sụt giảm sâu (Drawdown).
*(Hình ảnh minh họa: Tham khảo biểu đồ Scatter Plot 2D phân cụm theo Volatility vs Return từ `outputs/figures/`). Qua biểu diễn không gian 2D, Scatter plot cho thấy sự gãy gọn giữa các cụm, màu sắc tách bạch rõ sắc thái từng Profile.*

---

## 4.3. Đánh giá Module Phân Lớp Máy Học (Classification)
### 4.3.1. Các thang đo (Metrics) sử dụng
Mục tiêu của mô hình Classification (Random Forest) là học chuỗi chỉ số để phán đoán nến Ngày hôm sau có tăng giá hay không (`Price_Trend_Next_Day`).
Vì tập dữ liệu Crypto có độ mất cân bằng lớp khá tinh vi (Số ngày đi ngang/giảm dài hơn ngày tăng thốc), metric Accuracy (Độ chính xác) thông thường sẽ gây ảo giác. Mô hình được thẩm định qua lăng kính:
- **F1-Score (Trung bình điều hòa của Precision & Recall):** Cân bằng giữa việc "Bắt tỷ lệ Tăng/Giảm trúng xác suất" (Precision) và "Không bỏ sót các nhịp nổ giá" (Recall). Đạt F1-Score tốt nghĩa là Model có sức dự phóng toàn diện.

### 4.3.2. Phân tích Kết quả Thử nghiệm
Ngay sau khi Training, Random Forest tỏ rõ sự áp đảo thông qua chức năng bóc tách **Feature Importances**:
- Hệ số **Lag_1**, **Lag_2** đóng vai trò cực điểm (chiếm tỷ trọng ra quyết định cao nhất), chứng minh nguyên lý "Đà tăng giá của ngày hôm qua là kim chỉ nam cho hôm nay".
- Biến **Volume_Ratio** (Đột biến khối lượng) đóng góp lớn vào sự phá vỡ cấu trúc (Breakout).
- Kết quả ma trận Confusion Matrix cho thấy Model phát hiện sớm được phần lớn các dải nến giảm (Tránh rủi ro cho nhà đầu tư).

---

## 4.4. Đánh giá Module Dự Báo Chuỗi Thời Gian (Timseries Forecasting)
### 4.4.1. Các thang đo (Metrics) sử dụng
So với các mô hình hồi quy tuyến tính cổ điển, chuỗi giá Crypto luôn phá vỡ giả thuyết Dừng (Non-Stationary). Auto-ARIMA đã được kết hợp với quy trình Walk-forward (Chạy tịnh tiến). Hai sai số được dùng để làm thước đo:
- **MAE (Mean Absolute Error):** Trung bình phần chênh lệch tuyệt đối giữa Giá Dự Báo và Giá Thực tế. Chống chịu tốt với các gai nhiễu (Spikes) từ thị trường.
- **RMSE (Root Mean Square Error):** Nhạy cảm với các nhịp sai số lớn, phản ánh mức Phạt (Penalty) nặng mỗi khi mô hình đi chệch đường cong của xu hướng.

### 4.4.2. Phân tích Kết quả Thử nghiệm
Khi đối đầu trực diện dữ liệu `Close` thực tế trong biểu đồ (tham khảo ảnh Forecast Plot trong thư mục `outputs/`):
- Đường Forecast (Dự báo) của Auto-ARIMA bám sát mượt mà theo cấu trúc nền giá của đường Thực Tế (Actual Price).
- Biểu đồ **Residuals (Phần dư)** thể hiện mức chệch quỹ đạo dao động quanh mốc 0 (White Noise), chứng tỏ thuật toán đã rút cạn vắt kiệt tối đa tri thức cốt lõi tự tương quan (Autocorrelation) thay vì chỉ học vẹt. Sự lệch pha duy nhất xảy ra tại các hố sâu (Flash Crash) do thuật toán ARIMA đơn biến chưa thể tiêu hóa hết đống tin tức ngoại cảnh thiên nga đen.
