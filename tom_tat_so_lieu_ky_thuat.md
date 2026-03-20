# BẢNG TÓM TẮT SỐ LIỆU VÀ KỸ THUẬT LÕI (DÀNH CHO BÁO CÁO & TRẢ LỜI CÂU HỎI)

Tài liệu này tổng hợp toàn bộ các "Mảnh ghép Kỹ thuật" của hệ thống. Giúp anh/chị trả lời trơn tru các câu hỏi xoáy của Giảng viên: "Dùng thuật toán gì?", "Tại sao lại dùng nó?", "Đo lường bằng hệ số bao nhiêu?".

---

## 1. MODULE TIỀN XỬ LÝ & ĐẶC TRƯNG (Data Preprocessing)

| Vấn đề cần giải quyết | Phương pháp (Dùng cái gì?) | Lý do & Ý nghĩa (Cho cái gì?) | Số liệu / Tham số cài đặt |
| :--- | :--- | :--- | :--- |
| **Dữ liệu bị rỗng/thiếu (Missing Values)** | `Linear Interpolation` (Nội suy tuyến tính) + `bfill/ffill` | Trong chuỗi thời gian chứng khoán, giá không tự nhiên biến mất. Nội suy giúp nối lại đường gãy khúc của nến giá mà không làm méo mó xu hướng gốc. | 0% Missing Values sau khi chạy luồng Cleaner. |
| **Mô hình bị "mù" quá khứ** | Kéo trễ dữ liệu (`Lagging`): T-1, T-3, T-7 ngày | Ép các mô hình truyền thống (như Random Forest) phải có "trí nhớ" về động lượng giá của những ngày hôm trước. | Sinh ra 3 cột: `Lag_1`, `Lag_3`, `Lag_7`. |
| **Đo lường mức độ rủi ro (Risk)** | Tính Độ lệch chuẩn trượt (`Rolling Standard Deviation`) 7 ngày | Tiền điện tử biến động rất mạnh. Lấy phương sai 7 ngày gần nhất để định lượng chính xác "Độ xóc" của thị trường. | Cột sinh ra: `Volatility_7d` |

---

## 2. MODULE KHAI PHÁ LUẬT KẾT HỢP (Association Rules)

| Vấn đề cần giải quyết | Phương pháp (Dùng cái gì?) | Lý do & Ý nghĩa (Cho cái gì?) | Số liệu / Tham số cài đặt |
| :--- | :--- | :--- | :--- |
| **Tìm quy luật đồng pha (Nếu A tăng thì B có tăng không?)** | Thuật toán `Apriori` / `FP-Growth` | Quét qua rổ giao dịch của nhiều đồng coin cùng lúc để túm gọn tâm lý "bầy đàn" của thị trường. | Dữ liệu đầu vào: Danh sách Cắt lớp Trạng thái (Tăng/Giảm) của các Coin trong cùng 1 ngày. |
| **Đo lường độ Tin cậy của Luật** | Chỉ số `Confidence` (Tính bằng %) | Xác suất chắc chắn X xảy ra khi Y đã nổ. | **Ngưỡng lọc: > 0.5 (50%).** |
| **Bộ lọc chống "Trùng hợp ngẫu nhiên"** | Chỉ số `Lift` (Độ nâng) | Nếu Lift = 1, hai đồng tiền chỉ tình cờ tăng cùng nhau. Lift > 1 chứng tỏ chúng có sức hút nội tại kéo nhau lên. | **Ngưỡng lọc: Lift > 1.0.** Kết quả thực tế: Lúc thị trường hoảng loạn (High Volatility), Lift của nhóm `{Bitcoin_Down} -> {AltCoin_Down}` có thể lên mốc 2.0 hoặc 3.0. |

---

## 3. MODULE PHÂN CỤM DỮ LIỆU (Clustering)

| Vấn đề cần giải quyết | Phương pháp (Dùng cái gì?) | Lý do & Ý nghĩa (Cho cái gì?) | Số liệu / Tham số cài đặt |
| :--- | :--- | :--- | :--- |
| **Phân loại rổ tài sản đầu tư tự động** | Thuật toán `K-Means` | Trí tuệ AI tự do gom các đồng Coin có chung Tính cách (Hệ số Rủi ro & Tỷ suất Sinh lời) lại thành từng nhóm mà không cần con người dán nhãn trước. | Số cụm `K = 4` (Thường chia thành Nhóm An tòan, Nhóm Tiềm năng, Nhóm Bơm xả, Nhóm Chuẩn hóa). |
| **Tìm số lượng nhóm (K) tối ưu nhất** | Phương pháp `Elbow` (Khuỷu tay) đo bằng gia tốc `Inertia` | Tính tổng bình phương khoảng cách từ các điểm đến tâm cụm. Chỗ nào đồ thị bắt đầu gãy gập (bão hòa) là K đẹp nhất. | Dữ liệu mồi: `Volatility_7d` và `Close_Return`. |

---

## 4. MODULE PHÂN LỚP XU HƯỚNG TƯƠNG LAI (Classification)

| Vấn đề cần giải quyết | Phương pháp (Dùng cái gì?) | Lý do & Ý nghĩa (Cho cái gì?) | Số liệu / Tham số cài đặt |
| :--- | :--- | :--- | :--- |
| **Dự đoán Ngày mai Coin Tăng, Giảm hay Đứng im?** | Thuật toán `Random Forest Classifier` (Rừng quyết định) | Giá coin là chuỗi phi tuyến tính cực kỳ phức tạp. Random Forest sinh ra hàng trăm cây quyết định để chống bị học vẹt (Overfitting) và nhận diện được Cấu trúc ngầm của đà tăng giá. | Số lượng cây: `n_estimators = 100`.<br>Thông số: `class_weight='balanced'` (Để chống thiên vị vì số ngày Coin Rớt giá thường nhiều hơn Ngày Tăng). |
| **Mô hình bị lừa (Đánh giá sai lệch)** | Dùng metric `F1-Score` thay vì dùng `Accuracy` (Độ chính xác) | Nếu 90% thời gian Coin giảm giá, máy AI chỉ việc suốt ngày báo "Sẽ Giảm" là được 90% Accuracy. Nhưng F1-Score sẽ bắt lỗi và chấm điểm thấp ngay lập tức. | Kết quả chạy thực tế F1-score duy trì mức ổn định qua các tập Test chéo. |
| **Tìm ra đâu là gốc rễ đẩy giá tăng?** | Chức năng `Feature Importances` (Độ quan trọng Đặc trưng) | Xếp hạng xem Yếu tố Vĩ mô nào quyết định xu hướng tương lai. | Kết quả thực tế: Biến `Lag_1` (Quán tính ngày hôm qua) và `Volume_Ratio` (Sự bùng nổ Tỷ lệ Khối lượng giao dịch) nắm chi phối cao nhất (Khoảng > 30% trọng số). |

---

## 5. MODULE DỰ BÁO NHẢY SỐ (Timseries Forecasting)

| Vấn đề cần giải quyết | Phương pháp (Dùng cái gì?) | Lý do & Ý nghĩa (Cho cái gì?) | Số liệu / Tham số cài đặt |
| :--- | :--- | :--- | :--- |
| **Phóng chiếu chính xác giá Đóng cửa 14 ngày tới** | Thuật toán `Auto-ARIMA` kết hợp `Walk-Forward` (Cuốn chiếu) | Dữ liệu Crypto phá vỡ tính Dừng (Non-Stationary). Auto-ARIMA tự động rà quét lưới tham số lõi p,d,q để ép dữ liệu về trạng thái ổn định trước khi tiên tri, đồng thời tự sửa sai tịnh tiến qua từng ngày Test. | Dự báo xa: `Horizon = 14 days`. |
| **Đo lường độ lệch của đường dự báo** | Chỉ số `MAE` (Sai số Tuyệt đối Trung bình) và `RMSE` | Bắt lỗi phạt siêu nặng mỗi khi đường Dự báo phóng đi lệch quĩ đạo với đường Giá Thực Tế. | Nhìn vào biểu đồ `Residuals` (Dư lượng): Dư lượng dao động nhiễu trắng quanh mốc 0, chứng tỏ mô hình học vắt kiệt tối đa Tri thức tuyến tính. |
