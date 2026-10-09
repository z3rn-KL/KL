# Rà soát ba đầu việc backend KLTN_11 (09/10/2026)

## 1. Tích hợp phát hiện điểm bất lợi
- Bổ sung chế độ `depot_included=False` vào `screen_adverse_deliveries` cho ma trận đường đi chỉ gồm các điểm giao, không tự tạo kho giả.
- Bổ sung script `python/experiments/backend_completion_audit.py`: nối nhãn DBSCAN theo **delivery_id** vào ma trận khoảng cách đường đi có sẵn; xuất `audit_outputs/adverse_road_sample.csv`.
- Dataset mẫu: 15 điểm, có 1 điểm được đánh dấu cần xem xét; không suy ra chi phí vận hành tăng chỉ từ kết quả sàng lọc.
- **Giới hạn:** Chỉ là mẫu 15 điểm, chưa đánh giá đường đi toàn bộ 2394 đơn hàng. Khi ma trận không có depot, không tính chỉ số chi phí chèn tuyến (để trống).

## 2. Đối chiếu benchmark
- Bổ sung kiểm tra định danh workload, số đơn, SLA, quãng đường/thời gian không âm và phép tính tỷ lệ cải thiện với file `final_kmeans_guided_dqn_validation_paired.csv`.
- Đối chiếu 12 workloads từ 108 workloads trong manifest: **không phát hiện bất nhất số học ở các tiêu chí kiểm tra**.
- 6/12 workload thiếu `baseline_cluster_switches` do không lưu toàn bộ tuyến Vanilla DQN. Giữ nguyên giá trị thiếu, không suy diễn hoặc tự điền.
- **Chưa hoàn thành tái lập khoa học:** Chưa huấn luyện lại DQN và guided DQN trên cùng seed/workload; bảng CSV đúng phép tính không có nghĩa là kết quả mô hình đã được tái lập.

## 3. Graph OpenStreetMap
- Thêm `python/experiments/offline_graph_audit.py` để parse toàn bộ GraphML có sẵn (không cần mạng/OSMnx), đối chiếu ID node đã snap với CSV.
- Graph: 130939 nodes, 302803 edges; 2394 đơn hàng có 1309 node snap duy nhất, **1309/1309 node có mặt trong graph**; 302803 edges có `length` và `travel_time`.
- **Chưa kiểm chứng:** OSMnx load_graphml, nearest_node, shortest path của graph thực trên môi trường chuẩn; không có `osmnx` ở môi trường kiểm tra. Việc node có mặt trong graph không chứng minh tuyến đã đúng.

## Kiểm thử và bảo toàn dữ liệu
- `PYTHONPATH=python pytest -q python/tests --disable-warnings --maxfail=1`: **171 passed**.
- Không sửa dữ liệu gốc, các file `results/*.csv` cũ, thông số huấn luyện, model hay frontend.
- Đầu ra kiểm tra **mới** nằm tại `audit_outputs/` và có thể tái tạo bằng:
  - `PYTHONPATH=python python -m experiments.backend_completion_audit`
  - `PYTHONPATH=python python -m experiments.offline_graph_audit`
- Không thể tuyên bố cả ba đầu việc được xác nhận hoàn toàn. Đã hoàn thành tích hợp *offline trên mẫu*, kiểm tra số học của benchmark và kiểm toán graph, còn **retraining thực tế và thao tác OSMnx** cần chạy trong môi trường đầy đủ.
