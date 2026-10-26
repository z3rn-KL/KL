# Xây dựng hệ thống tối ưu tuyến giao hàng kết hợp kỹ thuật phân cụm và Reinforcement Learning trong logistics

## Giới thiệu

Đây là mã nguồn phục vụ khóa luận xây dựng hệ thống tối ưu tuyến giao hàng trong logistics bằng cách kết hợp:

- kỹ thuật phân cụm;
- thuật toán heuristic;
- Reinforcement Learning;
- mạng đường OpenStreetMap;
- FastAPI backend.

Hệ thống hỗ trợ:

- tiền xử lý dữ liệu giao hàng;
- phân cụm đơn hàng;
- tối ưu thứ tự giao hàng;
- đánh giá quãng đường và thời gian di chuyển;
- đánh giá SLA và độ trễ;
- phát hiện điểm giao hàng bất lợi;
- ước lượng chi phí vận hành;
- cung cấp API phục vụ frontend và demo.

---

## Dữ liệu

Dataset XeDu sau tiền xử lý gồm:

```text
2,394 deliveries
16 shippers
92 unique sender locations
105 deliveries missing expectedDeliveryTime
```

---

## Lập kế hoạch theo ngày: làm thêm giờ, dời đơn và gộp cụm nhỏ

Mô phỏng 30 ngày liên tiếp (27/06 đến 26/07/2024, 562 đơn thật) với ca làm 10 giờ,
10 phút phục vụ mỗi điểm, xe tối đa 100 kg và đơn tồn chuyển sang hôm sau (được ưu tiên).
Dữ liệu thật chỉ có trung bình 2.3 shipper hoạt động mỗi ngày, nên nhóm shipper cố định là 4 người;
các mức tải x2 đến x16 là kịch bản stress được lấy mẫu lại từ chính dữ liệu XeDu.
Các hệ số chi phí là tham số giả định, không phải chi phí thật của XeDu.

Ba chính sách khi vượt ca: luôn làm thêm giờ (`overtime`), luôn dời đơn (`defer`),
và lai (`hybrid`: làm thêm tối đa 2 giờ, đơn nào dời rẻ hơn làm thêm thì dời).
Hai cách phân bổ: mỗi cụm một shipper (`dedicated`) và gộp cụm dưới 3 đơn vào shipper cụm gần nhất
nếu vẫn xong trong ca (`merge_small`).

Kết quả chính (chi phí mỗi đơn, VND, phân bổ `dedicated`):

| Tải | overtime | defer | hybrid |
|---|---|---|---|
| x4 | 41,981 (3 ca vượt trần 2 giờ) | 43,499, tồn 11 đơn | 42,210, tồn 1 đơn |
| x12 | 37,950 (53 ca vượt trần) | 434,751, tồn 1,619 đơn | 183,105, tồn 898 đơn |

- `overtime` rẻ nhất nhưng không giới hạn giờ làm thêm, nên vi phạm trần 2 giờ.
  Báo cáo chỉ chọn "best" trong các chính sách tôn trọng trần: kết quả là `hybrid`.
- Dữ liệu thật (x1) và x2 không bao giờ vượt ca 10 giờ.
- Trên dữ liệu thật, gộp cụm nhỏ giảm chi phí khoảng 12% (99,253 xuống 87,378 VND/đơn)
  và dùng 2.97 thay vì 3.47 shipper, đổi lại đúng hạn giảm từ 100% xuống 99.5%.
- Khi tải vượt hẳn năng lực, thêm shipper hiệu quả hơn mọi chính sách: ở x12, 6 shipper
  đưa chi phí `hybrid` từ 183,105 xuống 57,502 VND/đơn.
- Hạn chế: mọi đơn được coi là phát sinh đầu ngày; `hybrid` chỉ so sánh với phạt dời đơn của
  một ngày tiếp theo; định tuyến dùng Nearest Neighbor kết hợp 2-opt.

Chạy lại:

```bash
cd python
python experiments/daily_planning_prepare_matrix.py   # ~46 giây, tạo results/daily_planning_road_matrix.npz
python experiments/daily_planning_experiment.py       # ~2 phút
```

Kết quả: `results/daily_planning_findings.txt`, `daily_planning_summary.csv`,
`daily_planning_daily.csv`, `daily_planning_sensitivity.csv`.
File ma trận `.npz` (25 MB) không được commit, dựng lại bằng lệnh đầu tiên.

---

## Proposed RL

Agent chấm điểm từng delivery ứng viên (Q-network 8 feature), reward bằng trừ biến thiên hàm mục tiêu J.
Quy trình: chia 76 train, 16 validation, 16 test; chọn mục tiêu và checkpoint chỉ bằng train và validation,
khóa trước khi đánh giá test.

Kết quả trên tập test (16 workload, 131 đơn), hàm mục tiêu J đã khóa (thấp hơn là tốt hơn):

| CW | DQN | Q-Learning | NN | SARSA | Proposed RL |
|---|---|---|---|---|---|
| 0.4257 | 0.4352 | 0.4495 | 0.4515 | 0.4575 | 0.4649 |

Proposed RL **không thắng** các baseline trên test. Trên validation nó chỉ ngang Nearest Neighbor
(J = 0.4688 so với 0.4640). Thử thêm 5 feature nhìn trước (`rl/proposed/lookahead_encoder.py`) cho
kết quả tệ hơn trên validation (J trung bình 0.515, 3 seed), nên không đưa vào đánh giá test
(`results/proposed_rl_lookahead_validation_findings.txt`).
Ở quy mô 5 đến 16 điểm mỗi workload, heuristic đã gần tối ưu và RL chưa có thêm lợi thế.

---

## API cho kết quả mới

```text
GET /api/results/daily-planning/summary      ?load_multiplier=&allocation=&overtime_policy=
GET /api/results/daily-planning/daily        ?load_multiplier=&allocation=&overtime_policy=   (bắt buộc cả ba)
GET /api/results/daily-planning/sensitivity  ?parameter=
GET /api/results/proposed-rl/final-test
```

`allocation` nhận `dedicated` hoặc `merge_small`; `overtime_policy` nhận `overtime`, `defer` hoặc `hybrid`.
`summary` trả thêm `best_within_cap`: chính sách rẻ nhất theo từng mức tải trong số những chính sách không vượt trần làm thêm giờ.
