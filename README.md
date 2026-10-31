# Tối ưu tuyến giao hàng: dữ liệu thật và giao diện web

Mã nguồn khóa luận trên dữ liệu giao hàng XeDu (TP.HCM) và mạng đường OpenStreetMap. Nhánh `feature/frontend-duy` gồm toàn bộ phần nền của `main` (phân cụm, Nearest Neighbor, Clarke-Wright, Q-Learning, SARSA, DQN, K-Means-guided DQN, API FastAPI) và lập kế hoạch theo ngày, cộng:

1. **Giao diện web** React (12 trang).
2. **Phân tích trên dữ liệu thật**: phát lại ca giao thật so với tối ưu.

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

## Lập kế hoạch theo ngày

Mô phỏng 30 ngày (562 đơn thật), ca 10 giờ, 10 phút phục vụ mỗi điểm, đơn chưa giao chuyển sang hôm sau. So sánh `overtime`, `defer`, `hybrid` (làm thêm tối đa 2 giờ) và hai cách phân bổ cụm (`dedicated`, `merge_small`). Mã ở `python/simulation/daily_planning.py`.

Dữ liệu thật không bao giờ vượt ca 10 giờ, nên các mức tải x4 đến x16 là **kịch bản dựng thêm**. Chi phí mỗi đơn (VND, `dedicated`): ở x12, `overtime` 37.950 nhưng 53 ca vượt trần 2 giờ, `defer` 434.751 (tồn 1.619 đơn), `hybrid` 183.105 (tồn 898 đơn). Chính sách tốt nhất trong trần làm thêm giờ là `hybrid`; gộp cụm nhỏ giảm khoảng 12% chi phí ở tải thật. Chạy lại: `python experiments/daily_planning_prepare_matrix.py` rồi `python experiments/daily_planning_experiment.py`.

## Giới hạn

- Hệ số tiền (lương, làm thêm giờ, phí, hoa hồng, mức phạt) là giả định; dữ liệu XeDu không có giá đơn hàng.
- Chi phí vận hành và các mức tải x4 đến x16 là kịch bản, không phải dữ liệu thật.
- Một số số liệu benchmark ở trang So sánh được ghi cứng trong `frontend/src/pages/Compare.tsx`, không đọc từ API.

Các nhánh khác: `main` (phần nền), `proposed-routing-rl` (mô tả và kết quả của Proposed RL, RL theo ca làm). Mã và trang web Proposed RL vẫn còn trong nhánh này, chỉ README không mô tả nữa.
