# Tối ưu tuyến giao hàng: Proposed RL, dữ liệu thật và giao diện web

Mã nguồn khóa luận trên dữ liệu giao hàng XeDu (TP.HCM) và mạng đường OpenStreetMap. Nhánh `feature/frontend-duy` gồm toàn bộ phần nền của `main` (phân cụm, Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN, K-Means-guided DQN, API FastAPI) và `proposed-routing-rl` (Proposed RL, lập kế hoạch theo ngày), cộng:

1. **Giao diện web** React (12 trang).
2. **Phân tích trên dữ liệu thật**: phát lại ca giao thật so với tối ưu.
3. **Đo khoảng trống cải thiện** của RL bằng nghiệm tối ưu chính xác.

## Dữ liệu và cách chạy

Dữ liệu và đồ thị đường **không nằm trong repo** (đã `.gitignore`): đặt `dataset/xedu/xedu_cleaned.csv` và `maps/xedu_drive.graphml` (166 MB). Kết quả đã tính nằm trong `results/`.

Cần Python 3.12 trở lên và Node.js 18 trở lên. 408 test.

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

Dữ liệu là đơn hàng thật. `GET /api/deliveries` **mặc định đã ẩn danh**: bỏ mã shipper và đường dẫn ảnh, cắt địa chỉ người gửi và người nhận còn đến cấp quận, làm tròn tọa độ 3 chữ số (khoảng 110 m). Việc ẩn danh chạy trước khi tìm kiếm, nên tham số `q` không dò ra được số nhà. Chỉ khi chạy máy cá nhân mới tắt bằng `XEDU_ANONYMIZE=0`; **không bao giờ tắt trên máy chủ công khai**. Mã đơn (`id`, `mdh`) và tên gói hàng vẫn được giữ vì giao diện dùng chúng.

Chưa có đăng nhập: trước khi deploy công khai cần đặt mật khẩu truy cập. Các endpoint `/api/routing` và `/api/evaluation` chưa được rà soát về dữ liệu trả về (chỉ giới hạn 30 đơn mỗi lần).

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

## RL theo ca làm (thưởng phạt mới)

Một episode là một ca làm của một shipper (8 đến 40 đơn khi huấn luyện, thử tới 50 đơn). Mỗi bước chọn một đơn hoặc dừng ca (các đơn còn lại dời sang hôm sau). Phần thưởng: hoa hồng 3% giá trị đơn cho đơn đúng hẹn (tối đa 10% nếu đến sớm), thưởng kết thúc ca sớm; phạt: thời gian và quãng đường, làm thêm giờ, đơn trễ, biên lợi nhuận dưới 30%, đơn dời. Phạt trễ luôn lớn hơn thưởng tối đa (cấu hình từ chối giá trị ngược lại). Mã ở `python/rl/shift/`. **Mọi hệ số tiền là giả định**: dữ liệu XeDu không có giá đơn.

Các phiên bản đã huấn luyện đầy đủ (chọn mô hình và seed bằng validation, rồi đánh giá một lần trên tập test thứ hai chưa dùng khi phát triển, 46 ca từ 10 đến 50 đơn), khoảng cách tới tham chiếu, thấp hơn là tốt hơn:

| Phương pháp | Tất cả 46 ca (%) | Chỉ ca có nghiệm tối ưu chính xác, 30 ca (%) |
|---|---|---|
| NN + local search | **2,94** | **2,71** |
| Attention, bắt chước rồi gradient chính sách | 6,60 ± 0,95 | 5,65 ± 0,20 |
| Bắt chước rồi gradient chính sách (MLP) | 8,79 ± 0,55 | 7,29 ± 0,25 |
| Attention, gradient chính sách từ đầu | 10,70 ± 0,05 | 10,74 ± 0,33 |
| Gradient chính sách từ đầu (MLP) | 13,25 ± 0,88 | 13,77 ± 0,58 |
| DQN (hai biến thể) | 59 đến 62 | 47 đến 52 |

"±" là độ lệch chuẩn mẫu giữa 2 đến 3 seed huấn luyện, nên chỉ mang tính tham khảo. Với ca trên 14 đơn, tham chiếu là kết quả local search nhiều lần, cùng họ phương pháp với NN + local search, nên cột bên phải (nghiệm chính xác) mới công bằng cho phép so sánh; thứ tự các phương pháp không đổi ở cả hai cột. Tập test này đã dùng để đánh giá, mọi thay đổi mô hình tiếp theo cần một tập test mới (ví dụ seed 404).

- **DQN không học được** (khoảng 50% trên validation sau 40.000 episode, mọi seed). **Gradient chính sách** học được, và mạng attention nhìn khoảng cách giữa các đơn còn lại tốt hơn MLP.
- Mô hình tốt nhất dùng **bắt chước heuristic làm điểm khởi đầu**, nên không phải RL thuần. RL thuần tốt nhất là gradient chính sách attention từ đầu (10,7%).
- So với NN + local search trên cùng 46 ca, mô hình chính thắng 10, hòa 6, thua 30. Ở ca 50 đơn RL tốt hơn NN + local search một lần chạy (3,04% so với 6,89%) nhưng không tốt hơn "NN dừng + local search" (0,45%); ở ca 30 đơn kết quả trái ngược giữa hai tập test, nên chưa kết luận được. Mỗi cỡ chỉ 4 đến 6 ca.
- Suy luận 5,6 ms mỗi ca, so với 0,21 giây của NN dừng + local search (khoảng 1 giây ở ca 50 đơn).

Tham chiếu: nghiệm tối ưu chính xác khi ca có tối đa 14 đơn, còn lại là nghiệm tốt nhất tìm được bằng local search nhiều lần (có thể bị vượt). Hai seed DQN gốc (45, 46) bị dừng sớm ở 10.000 episode để nhường bộ nhớ và không nằm trong đánh giá.

Chạy lại (mất nhiều giờ): `shift_rl_references.py`, `shift_rl_train.py`, `shift_make_demos.py`, `shift_pg_train.py`, rồi `shift_rl_final_test.py test2`. API: `GET /api/results/shift-rl`.

## Lập kế hoạch theo ngày

Mô phỏng 30 ngày (562 đơn thật), ca 10 giờ, 10 phút phục vụ mỗi điểm, đơn chưa giao chuyển sang hôm sau. So sánh `overtime`, `defer`, `hybrid` (làm thêm tối đa 2 giờ) và hai cách phân bổ cụm (`dedicated`, `merge_small`). Mã ở `python/simulation/daily_planning.py`.

Dữ liệu thật không bao giờ vượt ca 10 giờ, nên các mức tải x4 đến x16 là **kịch bản dựng thêm**. Chi phí mỗi đơn (VND, `dedicated`): ở x12, `overtime` 37.950 nhưng 53 ca vượt trần 2 giờ, `defer` 434.751 (tồn 1.619 đơn), `hybrid` 183.105 (tồn 898 đơn). Chính sách tốt nhất trong trần làm thêm giờ là `hybrid`; gộp cụm nhỏ giảm khoảng 12% chi phí ở tải thật. Chạy lại: `python experiments/daily_planning_prepare_matrix.py` rồi `python experiments/daily_planning_experiment.py`.

## Giới hạn

- Hệ số tiền (lương, làm thêm giờ, phí, hoa hồng, mức phạt) là giả định; dữ liệu XeDu không có giá đơn hàng.
- Chi phí vận hành và các mức tải x4 đến x16 là kịch bản, không phải dữ liệu thật.
- Kết quả Proposed RL (bản gốc) chỉ một seed, tập kiểm tra nhỏ (16 workload); các ca theo ca làm chỉ có 46 ca và 4 đến 6 ca mỗi cỡ lớn.
- Một số số liệu benchmark ở trang So sánh được ghi cứng trong `frontend/src/pages/Compare.tsx`, không đọc từ API.

Các nhánh khác: `main` (phần nền), `proposed-routing-rl` (Proposed RL và lập kế hoạch theo ngày).
