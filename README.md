# KLTN_11: Spatial clustering and Reinforcement Learning for delivery routing

Đề tài: Xây dựng hệ thống tối ưu tuyến giao hàng kết hợp kỹ thuật phân cụm và Reinforcement Learning trong logistics.

## Phạm vi repository

Đây là **mã nghiên cứu thuật toán và thực nghiệm Python**, chưa bao gồm frontend. Các module chính:

- `python/data/`: tải, làm sạch, chuẩn hóa và ánh xạ đơn hàng.
- `python/clustering/`: K-Means, DBSCAN, đánh giá cụm, kiểm tra ánh xạ và phát hiện điểm bất lợi.
- `python/routing/`: ma trận mạng đường, Nearest Neighbor, Clarke–Wright và các biến thể.
- `python/rl/`: môi trường State/Action/Reward, Q-Learning, SARSA và DQN.
- `python/evaluation/`, `python/analysis/`: chỉ tiêu tuyến và thống kê vận hành.
- `python/experiments/`: chương trình thực nghiệm; kết quả công bố là dữ liệu có sẵn tại `results/`.
- `python/tests/`: kiểm thử.

## Chuẩn bị môi trường

Nên sử dụng Python tương thích với các phiên bản thư viện trong `requirements.txt` trong môi trường ảo:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

`osmnx==2.1.1` cần thiết để tải/lưu GraphML, gán road node và bổ sung travel time. Các thao tác trên graph NetworkX có sẵn và bộ test không cần sử dụng OSMnx trực tiếp vẫn chạy được khi thiếu thư viện này.

## Kiểm thử

Chạy từ thư mục gốc dự án:

```bash
# Linux/macOS
PYTHONPATH=python python -m pytest -q python/tests
# Windows PowerShell
$env:PYTHONPATH="python"; python -m pytest -q python/tests
```

Kiểm tra cú pháp:

```bash
python -m compileall -q python
```

## Thực nghiệm

```bash
# Linux/macOS, chạy từ thư mục gốc
PYTHONPATH=python python -m experiments.kmeans_experiment
PYTHONPATH=python python -m experiments.dbscan_experiment
PYTHONPATH=python python -m experiments.nearest_neighbor_experiment
PYTHONPATH=python python -m experiments.clarke_wright_experiment
PYTHONPATH=python python -m experiments.kmeans_guided_dqn_sensitivity
```

**Cảnh báo:** Các script thực nghiệm có thể ghi đè CSV trong `results/`; hãy sao lưu thư mục trước khi chạy. Thử nghiệm DQN có thể tốn thời gian. Một số tác vụ dùng mạng đường tại `maps/xedu_drive.graphml` (dung lượng lớn) và dữ liệu đơn tại `dataset/xedu/`.

## Giải thích tích hợp clustering + RL

`kmeans_guided_dqn_sensitivity.py` sử dụng kết quả K-Means đã được xuất ra CSV, ghép theo `delivery_id`, rồi truyền nhãn tương ứng vào `RoutingEnvironment` qua `delivery_cluster_ids`. Reward phạt chuyển cụm (`cluster_switch_penalty`) là **hướng dẫn mềm (reward shaping)**, không phải tối ưu tuyến riêng độc lập trong từng cụm. `0.0` giữ reward cũ. Hiệu quả phải được kiểm chứng bằng các benchmark trên workload tương đương.

`screen_adverse_deliveries` trong `python/clustering/adverse_deliveries.py` là API độc lập để phân biệt điểm nhiễu DBSCAN với ứng viên bị cô lập theo mạng đường hoặc làm tăng độ dài chèn tuyến. Công thức insertion detour chỉ là **proxy**, không phải chi phí hoạt động thực tế; chưa có ngưỡng được hiệu chỉnh và chưa được tích hợp tự động vào tất cả experiment.

## Giới hạn kiểm chứng

- Các test nhỏ không thay thế kiểm chứng đường đi thực tế và huấn luyện lại RL trên dữ liệu đầy đủ.
- Các CSV có sẵn là kết quả nghiên cứu được cung cấp, **không phải kết quả đã được tái tạo trong lần kiểm thử hiện tại**.
- Thời gian đi đường là ước lượng, không phải giao thông thời gian thực.
- Repo không có frontend và chưa có API web độc lập.
- Khi sử dụng/so sánh số liệu, cần lưu cấu hình, seed, phiên bản dữ liệu và điều kiện đánh giá; không nên so sánh các kết quả khác điều kiện.

### Offline research verification (2026-10-09)

Run `PYTHONPATH=python python -m experiments.backend_completion_audit` to join the saved DBSCAN labels to the saved **15-delivery peer-only road matrix** and audit the stored paired DQN benchmark. Run `PYTHONPATH=python python -m experiments.offline_graph_audit` to check the cached graph XML and snapped node membership. Outputs are written under `audit_outputs/`. These checks do **not** retrain RL or execute OSMnx; consult `BAO_CAO_BA_VIEC_BACKEND_2026-10-09.md` for limitations.
