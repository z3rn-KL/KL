# TÀI LIỆU TỔNG KẾT & ĐỐI CHIẾU KHOA HỌC (WALKTHROUGH)
Hệ thống Tối ưu Tuyến giao hàng Logistics XeDu (TP.HCM) — KLTN_11

---

## 1. Bảng đối chiếu số liệu của Leader với File dữ liệu gốc

Bảng kiểm chứng dưới đây đối chiếu các tuyên bố trong báo cáo khoa học với dữ liệu lưu trữ gốc trong thư mục `results/`:

| Chỉ số khoa học | Tuyên bố của Leader | Giá trị trong file kết quả gốc | File kết quả tương ứng | Kết quả đối chiếu |
| :--- | :---: | :---: | :--- | :--- :--- |
| **Tỷ lệ đúng hạn SLA** | $100\%$ | $1{,}0$ ($100{,}0\%$) | `results/final_kmeans_guided_dqn_validation_paired.csv` | **Trùng khớp 100%** |
| **Mức đánh đổi quãng đường** | $+1{,}76\%$ | $+1{,}7639\%$ (dài hơn) | `results/final_validation_summary.json` (`ratio_of_means_distance_improvement_pct: -1.7639`) | **Trùng khớp 100%** |
| **Mức đánh đổi thời gian** | $+1{,}87\%$ | $+1{,}8714\%$ (chậm hơn) | `results/final_validation_summary.json` (`ratio_of_means_time_improvement_pct: -1.8714`) | **Trùng khớp 100%** |
| **Tỷ lệ đổi cụm (6 workload có baseline)** | Giảm từ $0{,}6201 \rightarrow 0{,}5778$ ($-6{,}83\%$) | Baseline: $0{,}62011$, Guided: $0{,}57777$ ($\Delta = -6{,}828\%$) | `results/final_validation_summary.json` & `results/final_kmeans_guided_dqn_validation_paired.csv` | **Trùng khớp 100%** |
| **Số lượng workload kiểm chứng** | 12 workload | 12 workload (`#101, #29, #103, #42, #104, #65, #106, #71, #105, #62, #97, #90`) | `results/final_kmeans_guided_dqn_validation_paired.csv` | **Trùng khớp 100%** |
| **Tổng số đơn hàng kiểm chứng** | 108 đơn hàng | Tổng cột `number_of_deliveries` = 108 đơn | `results/final_kmeans_guided_dqn_validation_paired.csv` | **Trùng khớp 100%** |
| **Quy mô tập dữ liệu XeDu** | 2.394 đơn, 16 shipper | 2.394 đơn, 16 shipper | `dataset/deliveries_cleaned.csv` | **Trùng khớp 100%** |
| **Độ nhạy tham số $\lambda$** | $\lambda \in \{0; 0{,}10; 0{,}25; 0{,}50\}$ | 4 giá trị $\lambda$ được đánh giá | `results/kmeans_guided_dqn_sensitivity_results.csv` | **Trùng khớp 100%** |

---

## 2. Danh mục Toàn bộ Endpoint API Backend (`python/api`)

### 2.1. Hệ thống & Giám sát sức khỏe
- `GET /`: Thông tin chung của API service.
- `GET /api/health`: Kiểm tra tính sẵn sàng của 6 file artifact dữ liệu cốt lõi (`dataset`, `road_graph`, `snapped_nodes`, `kmeans_results`, `dataset_benchmark`, `final_validation`).
- **Warm-up nền (`_warm_up_road_network` trong `python/api/main.py`)**: Nạp ngầm file đồ thị giao thông 166 MB (`hcm_road_network.graphml`) vào bộ nhớ cache thông qua thread daemon khi khởi động ứng dụng mà không chặn request.
- `GET /api/diagnostics/summary`: Báo cáo chẩn đoán trạng thái bộ nhớ, đồ thị đường và độ trễ.

### 2.2. Dữ liệu Đơn hàng (`deliveries`)
- `GET /api/deliveries`: Truy vấn phân trang 2.394 đơn hàng kèm lọc theo dịch vụ, SLA, cụm, sắp xếp.
- `GET /api/deliveries/stats`: Thống kê tổng hợp số đơn, số shipper, số đơn không hạn giao, phân bổ dịch vụ.
- `GET /api/deliveries/{delivery_id}`: Chi tiết một đơn hàng cụ thể theo mã ID.
- `GET /api/deliveries/workloads`: Danh sách các ca giao hàng (workloads) trong kịch bản thực tế.

### 2.3. Phân cụm Địa lý (`clustering`)
- `GET /api/clustering/kmeans`: Tải kết quả phân cụm K-Means chuẩn ($K=10$) trên 2.394 điểm giao.
- `POST /api/clustering/kmeans/run`: Chạy phân cụm K-Means tùy biến theo số cụm $K$ truyền vào.
- `POST /api/clustering/dbscan/run`: Chạy phân cụm mật độ DBSCAN (`eps_km`, `min_samples`) để lọc điểm bất lợi/nhiễu.
- `GET /api/clustering/quality`: Chỉ số đánh giá chất lượng phân cụm (Silhouette, Inertia, Davies-Bouldin).

### 2.4. Điểm bất lợi (`adverse-deliveries`)
- `GET /api/adverse-deliveries/outliers`: Lấy danh sách các điểm giao xa tâm cụm địa lý nhất theo phân vị.

### 2.5. Định tuyến Tuyến giao hàng (`routing`)
- `GET /api/routing/precomputed`: Danh mục các thuật toán RL đã lưu tuyến có sẵn (`kmeans_guided_dqn`, `dqn`, `sarsa`, `q_learning`).
- `GET /api/routing/precomputed/{algorithm}`: Trả về chi tiết thứ tự giao và hình học mạng đường OSM của tuyến RL đã huấn luyện.
- `POST /api/routing/optimize`: Tối ưu tuyến trực tiếp trên đồ thị đường bằng Nearest Neighbor hoặc Clarke–Wright.

### 2.6. Đánh giá & Chi phí (`evaluation` & `results`)
- `POST /api/evaluation/cost`: Ước lượng chi phí vận hành chuyến giao theo kịch bản chi phí đơn vị ($4.000\text{ ₫/km}$ nhiên liệu, $40.000\text{ ₫/giờ}$ nhân công, phụ phí trễ hạn).
- `GET /api/results/overview`: Tổng quan kết quả thực nghiệm toàn diện của hệ thống.
- `GET /api/results/summary`: Tổng hợp kết quả cốt lõi của đề tài (SLA, quãng đường, thời gian, đổi cụm).
- `GET /api/results/benchmark/summary`: Thống kê tổng hợp đối chuẩn các thuật toán trên tập workload.
- `GET /api/results/benchmark/runs`: Dữ liệu chi tiết từng lần chạy của các thuật toán trên từng workload (hỗ trợ lọc theo `algorithm` và `workload_id`).
- `GET /api/results/workloads`: Danh sách 108 workload kèm số lượng đơn hàng và thời gian đợt giao.
- `GET /api/results/sensitivity`: Dữ liệu khảo sát độ nhạy trọng số phạt $\lambda$.
- `GET /api/results/final-validation`: Dữ liệu kiểm chứng cặp đôi 12 workload giữa Guided DQN và Vanilla DQN.
- `GET /api/results/stability`: Dữ liệu kiểm tra độ ổn định qua 10 hạt giống (seeds) và độ nhạy 13 cấu hình siêu tham số.

---

## 3. Tổng hợp Các Thay đổi Theo File trong Dự án

| Tệp tin | Vị trí thay đổi chính | Mục đích và nội dung thay đổi |
| :--- | :--- | :--- |
| `frontend/src/App.tsx` | Routing & Nav shell | Khôi phục route `/do-nhay` thành trang riêng biệt (`Sensitivity.tsx`); đánh số thứ tự thanh điều hướng từ 1 đến 9; cập nhật lazy import. |
| `frontend/src/pages/Overview.tsx` | Stepper, Card KPI, Quy mô | Sửa bước 4 thành "giảm số lần đổi cụm"; card KPI hiển thị động `12 workload / 108 đơn` lấy từ API; sửa đường nối stepper không bị tràn khung. |
| `frontend/src/pages/RoutesPage.tsx` | Workload selector, Map, Chú giải | DQN Vanilla vẽ tuyến cho 11 workload kiểm chứng từ `baseline_delivery_order`; Q-Learning/SARSA ở workload khác #1 không tự nhảy về #1, hiển thị KPI từ `results/dataset_algorithm_all_results.csv` và thông báo không vẽ tuyến; ô chọn workload ghi chú "(có tuyến)" / "(chỉ có số liệu)"; chú giải tuyến so sánh nét đứt màu cam; chỉ 1 nhãn "Mạng đường thực OSM"; tiêu đề khối chi phí/hiệu năng gắn tên thuật toán; không tô xanh số reward; bỏ câu thừa. |
| `frontend/src/components/MapBase.tsx` | MapContainer | Thêm `zoomControl={false}` để sửa ô trắng ở góc trên trái bản đồ. |
| `frontend/src/pages/Compare.tsx` | Table & Footnote | Thêm chú thích Q-Learning và SARSA có số liệu tổng cho 12 workload nhưng không có dữ liệu tuyến chi tiết; chuẩn hóa delta $\Delta = G - V$. |
| `frontend/src/pages/Sensitivity.tsx` | Toàn bộ trang | Phục hồi thành trang riêng biệt hiển thị khảo sát độ nhạy $\lambda$ và bảng kiểm chứng cặp đôi 12 workload. |
| `frontend/src/pages/Stability.tsx` | Bảng & Card | Cập nhật thông số 10 seeds, 5.000 episodes/seed; định dạng số học chuẩn tiếng Việt. |
| `frontend/src/pages/Clusters.tsx` | Silhouette, Tách nhiễu DBSCAN | Tách riêng điểm nhiễu DBSCAN và điểm xa tâm K-Means; hiển thị Silhouette score chuẩn. |
| `README.md` | Bảng ánh xạ, Giới hạn | `/do-nhay` là trang riêng; Stability 10 seed, 5.000 episode; bảng trang 1-9 khớp sidebar; bổ sung đầy đủ 5 giới hạn đã biết. |
| `docs/WALKTHROUGH.md` | Toàn bộ tài liệu | Tài liệu tổng kết đối chiếu toàn diện số liệu leader, endpoint API, thay đổi file, hiệu ứng, bundle size và kết quả test. |

---

## 4. Danh sách Hiệu ứng Giao diện (Micro-Interactions)
1. **Bản đồ luồng di chuyển (Dash Animation Flow)**: Tuyến đường cyan phát sáng với dòng chảy nét đứt chuyển động theo chiều di chuyển của shipper.
2. **Kho hàng nhấp nháy nhẹ (Pulsing Depot)**: Hiệu ứng tỏa sáng nhẹ chu kỳ 2.4s quanh điểm xuất phát để người dùng dễ định vị.
3. **Ghim giao hàng xuất hiện tuần tự (Sequential Stop Pins)**: Ghim điểm nhận hàng hiện dần theo tiến độ xe lăn bánh trên tuyến.
4. **Hiệu ứng sóng chuyển cụm (Cluster Ripple)**: Đánh dấu vị trí đổi cụm bằng hiệu ứng gợn sóng màu vàng hổ phách.
5. **Stepper 5 bước tự tô đầy khi cuộn (Scroll-triggered Stepper)**: Thanh nối nét đứt tự động tô màu khi cuộn qua từng phần câu chuyện đề tài.
6. **Vòng quay tiến độ SLA Gauge**: SVG gauge vẽ vòng tròn mượt mà khi nạp số liệu đúng hạn 100%.
7. **Thanh phân kỳ chênh lệch % (Diverging Bars)**: Minh họa trực quan độ dài đánh đổi và mức giảm đổi cụm.
8. **Modal Command Palette Blur & Scale**: Backdrop phủ mờ `blur(6px)`, hộp thoại phóng nhẹ `scale(0.97) -> scale(1)`.
9. **Tôn trọng `prefers-reduced-motion`**: Mọi hiệu ứng chuyển động tự động tắt hoặc fallback về hiển thị tức thì khi người dùng bật chế độ giảm chuyển động trong hệ điều hành hoặc qua công tắc giao diện.

---

## 5. Thống kê Kích thước Bundle Frontend (Production Build)

Kết quả lệnh `npm run build` (`tsc --noEmit && vite build`):
- `dist/index.html`: **1.49 KB** (gzip: **0.77 KB**)
- `dist/assets/index-*.css`: **53.8 KB** (gzip: **15.2 KB**)
- `dist/assets/index-*.js` (App shell + Command Palette): **139.0 KB** (gzip: **43.2 KB**)
- `dist/assets/vendor-*.js` (React, React-DOM, Router): **163.7 KB** (gzip: **53.4 KB**)
- `dist/assets/leaflet-*.js` (Leaflet engine): **155.1 KB** (gzip: **45.3 KB**)
- `dist/assets/charts-*.js` (Recharts SVG engine): **424.1 KB** (gzip: **112.5 KB**)
- `dist/assets/Sensitivity-*.js` (Trang Độ nhạy & Kiểm chứng): **10.5 KB** (gzip: **3.2 KB**)
- **Thời gian build**: $\approx 4{,}0\text{ giây}$.
- **Tổng dung lượng mạng tải trang đầu (Initial Gzip)**: $\approx 112\text{ KB}$ (chỉ tải shell và trang hiện tại nhờ lazy loading toàn diện).

---

## 6. Kết quả Thực thi Kiểm thử Hệ thống

1. **Frontend TypeCheck & Build (`npm run build`)**:
   - TypeScript strict check: **0 lỗi**
   - Vite bundle: **Thành công 100%**
2. **Frontend Automated Smoke Test (`npm test`)**:
   - `9/9` trang kiểm thử: **Passed 100%**
   - Kiểm thử Command Palette (`Ctrl+K`): **Passed**
   - Lỗi Console / Runtime: **0 lỗi**
3. **Backend API Pytest (`pytest python/tests/`)**:
   - `7/7` test cases: **Passed 100%**

---

## 7. Các Hạn chế Chưa Làm Được (Remaining Gaps)

1. **Chưa thực hiện huấn luyện lại trực tiếp từ đầu trên toàn bộ 108 workload**:
   - Do mỗi ca huấn luyện sâu (Deep RL) 5.000 episodes đòi hỏi nhiều giờ tính toán GPU, hệ thống hiện hiển thị kết quả kiểm chứng trên 12 workload đại diện đã được ghi nhận trong `final_validation_results`.
2. **Thiếu dữ liệu tuyến gốc Vanilla DQN cho 6 workload để so sánh tỷ lệ đổi cụm**:
   - File kết quả gốc chỉ lưu chi tiết chuỗi cụm của mô hình Vanilla DQN ở 6/12 workload trong thí nghiệm độ nhạy $\lambda = 0{,}0$. Hệ thống giữ nguyên trạng thái thiếu dữ liệu này cho 6 workload còn lại mà không nội suy giả định.
3. **Khả năng hoạt động ngoại tuyến của bản đồ**:
   - Do bản đồ sử dụng máy chủ tile trực tuyến miễn phí của OpenStreetMap, nếu không có mạng Internet người dùng sẽ không thấy các lớp đường phố chi tiết bên dưới, dù tuyến đường và tọa độ các điểm vẫn được vẽ chính xác.
4. **Dữ liệu thứ tự điểm dừng cho Q-Learning và SARSA**:
   - Nghiên cứu chỉ lưu chuỗi điểm dừng chi tiết của Q-Learning và SARSA tại Workload #1; trên 11 workload kiểm chứng còn lại chỉ có số liệu KPI tổng hợp mà chưa có dữ liệu hành trình để vẽ đa giác tuyến trên bản đồ.

