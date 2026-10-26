# Frontend – KLTN_11

Giao diện web (React + Vite + TypeScript) minh họa hệ thống tối ưu tuyến giao
hàng kết hợp phân cụm và Reinforcement Learning. Frontend chỉ gọi API FastAPI
trong `python/api`, không tự tính toán thuật toán.

## Các trang

| Trang | Đường dẫn | Nội dung | Endpoint chính |
|---|---|---|---|
| Tổng quan | `/tong-quan` | Số liệu dữ liệu, quy trình, KPI benchmark, cơ cấu dịch vụ | `/api/results/overview`, `/api/deliveries/stats` |
| Đơn hàng | `/don-hang` | Danh sách đơn có phân trang, tìm kiếm, lọc (dịch vụ, đúng hạn/trễ, cụm), sắp xếp, chi tiết đơn, vị trí trên bản đồ | `/api/deliveries` |
| Phân cụm & điểm bất lợi | `/phan-cum` | K-Means luận văn (K=10), chạy lại K-Means/DBSCAN, điểm nhiễu, điểm xa tâm cụm | `/api/clustering/*` |
| Tuyến giao hàng | `/tuyen-giao` | Tuyến Q-Learning/SARSA/DQN/DQN+K-Means trên bản đồ, chi phí, so sánh theo workload, chạy NN/Clarke–Wright | `/api/routing/*`, `/api/results/benchmark/runs` |
| Chạy thử thuật toán | `/chay-thu` | Chọn tập đơn (mẫu theo cụm / workload / thủ công, tối đa 30), chạy NN và Clarke–Wright trực tiếp, xem đầu vào, tuyến theo đường phố, chi phí, so sánh | `/api/routing/optimize`, `/api/evaluation/cost`, `/api/deliveries` |
| So sánh thuật toán | `/so-sanh` | Bảng và biểu đồ 5 thuật toán (có cột chi phí vận hành), thắng/hoà/thua so với NN, ảnh hưởng quy mô | `/api/results/benchmark/*`, `/api/evaluation/cost` |
| Ổn định & chọn tham số | `/on-dinh` | Silhouette/elbow chọn K, lưới DBSCAN, ổn định Q-Learning theo seed và siêu tham số | `/api/results/clustering-quality`, `/api/results/stability` |
| Độ nhạy & kiểm chứng | `/do-nhay` | Hệ số phạt đổi cụm λ, kiểm chứng cặp đôi 12 workload | `/api/results/sensitivity`, `/api/results/final-validation` |

## Chạy thử

Cần Node.js 18+ và backend đang chạy ở cổng 8000.

```bash
# Terminal 1 – backend (từ thư mục gốc repo)
cd python
PYTHONPATH=. uvicorn api.main:app --port 8000

# Terminal 2 – frontend
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite proxy mọi yêu cầu `/api/*` tới backend. Nếu backend chạy ở địa chỉ khác:

```bash
VITE_API_TARGET=http://host:port npm run dev
```

Build sản phẩm: `npm run build` (kết quả trong `dist/`), xem thử bằng
`npm run preview` (cũng có proxy `/api`).

## Dữ liệu cần có ở máy chạy backend

- `dataset/xedu/xedu_cleaned.csv`: cần cho trang Đơn hàng và thông tin điểm giao
  ở trang Tuyến giao hàng.
- `maps/xedu_drive.graphml`: cần để vẽ tuyến theo đường phố và để nút “Chạy thêm
  NN & Clarke–Wright” hoạt động. Nếu thiếu, trang tuyến vẫn hiển thị thứ tự giao
  (nối các điểm theo đường thẳng) kèm thông báo.
- Các file trong `results/` (đã có trong repo) đủ cho Tổng quan, So sánh, Độ
  nhạy và bản đồ phân cụm K-Means.

Trạng thái backend hiện ở góc dưới thanh bên trái (lấy từ `/api/health`).

## Backend: các thay đổi đi kèm

- `GET /api/deliveries` nhận thêm `q`, `service_type`, `is_late`, `cluster_id`,
  `sort_by`, `sort_dir`; mọi đơn nay có `cluster_id` (nối từ K-Means).
- Mới: `GET /api/results/stability`, `GET /api/results/clustering-quality`
  (`python/api/routers/diagnostics.py`, chỉ đọc CSV có sẵn trong `results/`).
- Backend nạp trước đồ thị đường (166 MB) ở luồng nền khi khởi động, nên yêu cầu
  tuyến đầu tiên không phải chờ nạp lại. Lần chạy tập đơn mới vẫn có thể mất
  vài chục giây để tính ma trận đường đi; chạy lại cùng tập thì nhanh.
- Test mới: `python/tests/test_api_diagnostics.py`.

## Lưu ý khi đọc kết quả

- Trang “Chạy thử”: tỷ lệ đúng hạn chỉ có ý nghĩa khi các đơn thuộc cùng một
  đợt giao. Với mẫu theo cụm hoặc chọn thủ công, các đơn thường tạo cách nhau
  nhiều ngày nên mốc bắt đầu tự suy ra làm số phút trễ rất lớn; hãy dùng
  “Workload benchmark” hoặc nhập mốc bắt đầu.
- Dữ liệu ổn định Q-Learning chỉ từ một workload 5 đơn (có nghiệm vét cạn),
  không đại diện cho quy mô toàn dataset.

- “Ứng viên điểm bất lợi” ở trang phân cụm là chỉ báo khoảng cách địa lý tới tâm
  cụm, chưa phải chi phí chèn tuyến trên mạng đường (chức năng đó nằm ở
  `screen_adverse_deliveries` và cần ma trận đường đi).
- Tuyến RL là tuyến đã huấn luyện sẵn; API không huấn luyện lại khi được gọi.
- Chi phí vận hành là kịch bản mô phỏng của luận văn, không phải số liệu kế toán.
- Kho mặc định (10.7769, 106.7009) lấy theo `simulation/scenario.py`; đổi tại
  `src/lib/format.ts` nếu kịch bản thay đổi.
