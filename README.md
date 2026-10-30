# Tối ưu tuyến giao hàng: Proposed RL, dữ liệu thật và giao diện web

Mã nguồn khóa luận trên dữ liệu giao hàng XeDu (TP.HCM) và mạng đường OpenStreetMap. Nhánh `feature/frontend-duy` gồm toàn bộ phần nền của `main` (phân cụm, Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN, K-Means-guided DQN, API FastAPI) và `proposed-routing-rl` (Proposed RL, lập kế hoạch theo ngày), cộng:

1. **Giao diện web** React (12 trang).
2. **Phân tích trên dữ liệu thật**: phát lại ca giao thật so với tối ưu.
3. **Đo khoảng trống cải thiện** của RL bằng nghiệm tối ưu chính xác.

## Dữ liệu và cách chạy

Dữ liệu và đồ thị đường **không nằm trong repo** (đã `.gitignore`): đặt `dataset/xedu/xedu_cleaned.csv` và `maps/xedu_drive.graphml` (166 MB). Kết quả đã tính nằm trong `results/`.

Cần Python 3.12 trở lên và Node.js 18 trở lên. 378 test.

```bash
pip install -r requirements.txt
pytest python                                    # từ thư mục gốc (một số test dùng đường dẫn results/)

# Terminal 1: backend
cd python && PYTHONPATH=. uvicorn api.main:app --port 8000

# Terminal 2: giao diện, http://localhost:5173 (proxy /api sang cổng 8000)
cd frontend && npm install && npm run dev
```

Build giao diện: `npm run build`. Chi tiết các trang và endpoint: `frontend/README.md`.

## Quyền riêng tư trước khi deploy

Dữ liệu là đơn hàng thật. `GET /api/deliveries` trả địa chỉ người gửi và người nhận, tọa độ, mã shipper, tên gói hàng và đường dẫn ảnh. **Chưa có cơ chế ẩn danh hay đăng nhập.** Trước khi đưa lên web công khai cần ẩn các trường này (hoặc chỉ phục vụ số tổng hợp) và đặt mật khẩu truy cập. Các trang chỉ dùng số tổng hợp (Dữ liệu thật, Kế hoạch ngày, Proposed RL) không lộ dữ liệu cá nhân.

## Phân tích trên dữ liệu thật

Đầu vào là thật: đơn, tọa độ, hạn giao, thời điểm giao thật, đồ thị OSM. Quãng đường và thời gian của tuyến được tính trên đồ thị với tốc độ lý tưởng, **không đo ngoài đời**, và chưa thuật toán nào được thử thực tế.

- **Thực tế giao trễ 20,3%** (2.281 đơn có hạn giao).
- **Mô hình thời gian lạc quan hơn thực tế nhiều:** giữa hai lần giao liên tiếp, thực tế trung vị 28,9 phút, mô hình đường phố 6,9 phút (chênh trung vị 20,3 phút mỗi điểm cho phục vụ, lấy hàng, chờ, kẹt xe). Vì vậy SLA 96 đến 100% ở bảng benchmark là số tính theo mô hình, không phải thực tế.
- **Phát lại 64 ca sạch** (một shipper, một điểm lấy hàng dùng làm kho, mọi đơn tạo trước khi giao; 3 đến 7 đơn, trung vị 3):

| | Dài hơn tối ưu (thời gian) | Dài hơn tối ưu (quãng đường) |
|---|---|---|
| Thứ tự giao thật | +10,8% (CI 95%: 7,5 đến 14,5) | +11,6% |
| Nearest Neighbor | +3,7% | +4,6% |
| NN + local search | +0,16% | +0,10% |

Thứ tự giao thật không bám theo hạn giao (16/64 ca đúng thứ tự hạn sớm nhất). Tối ưu là cận dưới cho bài toán tĩnh nên khoảng cách tới nó không phải mức tiết kiệm có thể hứa. Mẫu nhỏ và đã chọn lọc.

Chạy lại: `python experiments/real_data_replay.py` (khoảng 20 giây, cần dữ liệu, đồ thị và ma trận đường). Kết quả: `results/real_replay_*`. API: `GET /api/results/real-data`.

## Proposed RL

Q-network nhỏ chấm điểm từng đơn ứng viên (8 đặc trưng, hai lớp ẩn 64, huấn luyện kiểu DQN), phần thưởng là trừ phần tăng thêm của hàm mục tiêu J. 76 workload huấn luyện, 16 kiểm định, 16 kiểm tra; mô hình khóa trước khi chạm tập kiểm tra. Mã ở `python/rl/proposed/`.

Trên tập kiểm tra (16 workload, 131 đơn), J càng thấp càng tốt:

| CW | DQN | Q-Learning | NN | SARSA | Proposed RL |
|---|---|---|---|---|---|
| 0,4257 | 0,4352 | 0,4495 | 0,4515 | 0,4575 | 0,4649 |

**Proposed RL không thắng các thuật toán đối chứng.** Thêm 5 đặc trưng nhìn trước cho kết quả tệ hơn nên không đưa sang kiểm tra.

**Khoảng trống cải thiện** (16 workload kiểm định, 5 đến 13 điểm, nghiệm tối ưu chính xác bằng quy hoạch động, đã kiểm chứng với vét cạn): Proposed RL cách tối ưu 8,7%, Nearest Neighbor 7,6%, **NN + local search chỉ 0,49%** (`results/proposed_rl_headroom_validation_findings.txt`). Một hàm điểm theo ca làm với phần thưởng hoa hồng và phạt trễ (`python/rl/shift/`, tiền là giả định) cho khoảng cách 2,2% ở ca đến 14 đơn (`results/shift_headroom_validation_findings.txt`). Kết luận: ở quy mô ca thật (3 đến 13 đơn), heuristic có local search gần như tối ưu nên RL khó có chỗ cải thiện; hướng còn lại là điều phối trực tuyến khi đơn đến dần, chưa làm.

## Lập kế hoạch theo ngày

Mô phỏng 30 ngày (562 đơn thật), ca 10 giờ, 10 phút phục vụ mỗi điểm, đơn chưa giao chuyển sang hôm sau. So sánh `overtime`, `defer`, `hybrid` (làm thêm tối đa 2 giờ) và hai cách phân bổ cụm (`dedicated`, `merge_small`). Mã ở `python/simulation/daily_planning.py`.

Dữ liệu thật không bao giờ vượt ca 10 giờ, nên các mức tải x4 đến x16 là **kịch bản dựng thêm**. Chi phí mỗi đơn (VND, `dedicated`): ở x12, `overtime` 37.950 nhưng 53 ca vượt trần 2 giờ, `defer` 434.751 (tồn 1.619 đơn), `hybrid` 183.105 (tồn 898 đơn). Chính sách tốt nhất trong trần làm thêm giờ là `hybrid`; gộp cụm nhỏ giảm khoảng 12% chi phí ở tải thật. Chạy lại: `python experiments/daily_planning_prepare_matrix.py` rồi `python experiments/daily_planning_experiment.py`.

## Giới hạn

- Hệ số tiền (lương, làm thêm giờ, phí, hoa hồng, mức phạt) là giả định; dữ liệu XeDu không có giá đơn hàng.
- Chi phí vận hành và các mức tải x4 đến x16 là kịch bản, không phải dữ liệu thật.
- Kết quả Proposed RL chỉ một seed, tập kiểm tra nhỏ (16 workload).
- Một số số liệu benchmark ở trang So sánh được ghi cứng trong `frontend/src/pages/Compare.tsx`, không đọc từ API.

Các nhánh khác: `main` (phần nền), `proposed-routing-rl` (Proposed RL và lập kế hoạch theo ngày).
