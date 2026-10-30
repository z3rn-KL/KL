# Tối ưu tuyến giao hàng: Proposed RL và lập kế hoạch theo ngày

Mã nguồn khóa luận trên dữ liệu giao hàng XeDu (TP.HCM) và mạng đường OpenStreetMap. Nhánh `proposed-routing-rl` gồm toàn bộ phần nền của `main` (phân cụm, Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN, K-Means-guided DQN, API FastAPI) cộng hai phần mới:

1. **Proposed RL**: agent chấm điểm từng đơn ứng viên để chọn đơn giao tiếp theo.
2. **Lập kế hoạch theo ngày**: làm thêm giờ hay dời đơn, gộp cụm ít đơn.

## Dữ liệu và cách chạy

Dữ liệu và đồ thị đường **không nằm trong repo** (đã `.gitignore`): đặt `dataset/xedu/xedu_cleaned.csv` và `maps/xedu_drive.graphml`. Dữ liệu có địa chỉ khách thật, không đưa lên nơi công khai. Kết quả đã tính nằm trong `results/`.

Python 3.12 trở lên, 319 test.

```bash
pip install -r requirements.txt
pytest python                                    # từ thư mục gốc
cd python && PYTHONPATH=. uvicorn api.main:app --port 8000
```

## Proposed RL

Q-network nhỏ (8 đặc trưng của ứng viên, hai lớp ẩn 64), huấn luyện kiểu DQN. Phần thưởng mỗi bước là trừ phần tăng thêm của hàm mục tiêu J (quãng đường, thời gian, đơn trễ, phút trễ, số lần đổi cụm). Mã ở `python/rl/proposed/`.

Quy trình: 76 workload huấn luyện, 16 kiểm định, 16 kiểm tra. Hàm mục tiêu và checkpoint chọn trên huấn luyện và kiểm định, **khóa trước khi chạm vào tập kiểm tra**.

Kết quả trên tập kiểm tra (16 workload, 131 đơn), J càng thấp càng tốt:

| CW | DQN | Q-Learning | NN | SARSA | Proposed RL |
|---|---|---|---|---|---|
| 0,4257 | 0,4352 | 0,4495 | 0,4515 | 0,4575 | 0,4649 |

**Proposed RL không thắng các thuật toán đối chứng.** Trên kiểm định nó chỉ ngang Nearest Neighbor (J = 0,4688 so với 0,4640). Thử thêm 5 đặc trưng nhìn trước cho kết quả tệ hơn (J trung bình 0,515 trên 3 seed) nên không đưa sang kiểm tra (`results/proposed_rl_lookahead_validation_findings.txt`). Mỗi workload chỉ có 5 đến 16 điểm nên heuristic đã gần tối ưu; đây là nhận định từ kết quả đo.

## Lập kế hoạch theo ngày

Mô phỏng 30 ngày liên tiếp (562 đơn thật), ca 10 giờ, 10 phút phục vụ mỗi điểm, đơn chưa giao chuyển sang hôm sau. So sánh ba chính sách khi hết ca (`overtime`, `defer`, `hybrid` làm thêm tối đa 2 giờ) và hai cách phân bổ cụm (`dedicated`, `merge_small`). Mã ở `python/simulation/daily_planning.py`.

Dữ liệu thật không bao giờ vượt ca 10 giờ, nên các mức tải x4 đến x16 là **kịch bản dựng thêm** từ chính dữ liệu. Chi phí mỗi đơn (VND, `dedicated`):

| Tải | `overtime` | `defer` | `hybrid` |
|---|---|---|---|
| x4 | 41.981 (3 ca vượt trần 2 giờ) | 43.499, tồn 11 đơn | 42.210, tồn 1 đơn |
| x12 | 37.950 (53 ca vượt trần) | 434.751, tồn 1.619 đơn | 183.105, tồn 898 đơn |

Dời đơn hoàn toàn tệ nhất khi quá tải. `overtime` rẻ nhất nhưng vượt trần 2 giờ, nên chọn tốt nhất trong các chính sách không vượt trần là `hybrid`. Gộp cụm nhỏ giảm khoảng 12% chi phí ở tải thật (99.253 xuống 87.378 VND mỗi đơn).

Chạy lại: `python experiments/daily_planning_prepare_matrix.py` (khoảng 46 giây, tạo ma trận đường, không commit) rồi `python experiments/daily_planning_experiment.py`.

## API thêm

| Endpoint | Nội dung |
|---|---|
| `GET /api/results/daily-planning/summary` | Tổng hợp các chính sách, kèm `best_within_cap` |
| `GET /api/results/daily-planning/daily` | Đơn tồn theo ngày (cần cả `load_multiplier`, `allocation`, `overtime_policy`) |
| `GET /api/results/daily-planning/sensitivity` | Độ nhạy theo tham số |
| `GET /api/results/proposed-rl/final-test` | Kết quả kiểm tra cuối của Proposed RL |

## Giới hạn

- Hệ số chi phí (lương, làm thêm giờ, mức phạt) là giả định; dữ liệu XeDu không có giá đơn hàng.
- Mọi đơn coi như phát sinh đầu ngày; `hybrid` chỉ tính mức phạt dời đơn của một ngày; định tuyến dùng Nearest Neighbor kết hợp 2-opt.
- Thời gian và SLA ở phần nền là tính theo mô hình đường phố; thực tế khoảng 20% đơn giao trễ.
- Kết quả cuối chỉ một seed, tập kiểm tra nhỏ.

Phân tích trên dữ liệu thật và giao diện web nằm ở nhánh `feature/frontend-duy`.
