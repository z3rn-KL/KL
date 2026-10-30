# Tối ưu tuyến giao hàng: phân cụm và học tăng cường

Mã nguồn khóa luận. Hệ thống tối ưu thứ tự giao hàng trên dữ liệu giao hàng XeDu (TP.HCM) và mạng đường OpenStreetMap, kết hợp phân cụm, heuristic và học tăng cường. Backend FastAPI.

Nhánh `main` chứa phần nền đã hoàn thành. Phần Proposed RL, lập kế hoạch theo ngày và giao diện web nằm ở các nhánh khác (xem cuối).

## Có gì trong nhánh này

- Tiền xử lý, phân cụm K-Means (K = 10) và DBSCAN, phát hiện điểm giao bất lợi.
- Định tuyến: Nearest Neighbor, Clarke-Wright (trên đường phố thật), Q-Learning, SARSA, DQN, K-Means-guided DQN.
- Đánh giá quãng đường, thời gian, SLA, độ trễ và chi phí vận hành ước lượng.
- API FastAPI (`python/api`) và 235 test.

## Dữ liệu

2.394 đơn, 16 shipper, 92 điểm lấy hàng, 105 đơn thiếu hạn giao.

Dữ liệu và đồ thị đường **không nằm trong repo** (đã `.gitignore`): đặt `dataset/xedu/xedu_cleaned.csv` và `maps/xedu_drive.graphml` (166 MB). Dữ liệu có địa chỉ khách hàng thật, không đưa lên nơi công khai. Các kết quả đã tính nằm trong `results/`.

## Chạy

Python 3.12 trở lên.

```bash
pip install -r requirements.txt
pytest python                                    # từ thư mục gốc
cd python && PYTHONPATH=. uvicorn api.main:app --port 8000
```

Tài liệu API ở `/docs`, trạng thái dữ liệu ở `/api/health`.

## Kết quả chính

Tính trên đồ thị đường, trung bình mỗi đợt giao, 108 đợt (813 đơn):

| Thuật toán | Quãng đường (km) | Thời gian (phút) | Đúng hạn |
|---|---|---|---|
| Nearest Neighbor | 62,68 | 68,13 | 99,41% |
| Clarke-Wright | 58,05 | 63,27 | 99,33% |
| Q-Learning | 62,26 | 67,69 | 100% |
| SARSA | 62,86 | 68,31 | 99,86% |
| DQN | 59,83 | 65,13 | 100% |

K-Means-guided DQN, trên 12 đợt kiểm chứng: ngắn hơn Nearest Neighbor 5,49% nhưng dài hơn Clarke-Wright 3,96% và DQN thường 1,76%; giảm số lần đổi cụm 6,83% (6 đợt). Kết quả tái hiện được với cùng seed.

## Giới hạn

- Thời gian và SLA là **tính theo mô hình**: tốc độ lý tưởng trên đồ thị OSM, không tính thời gian phục vụ tại điểm giao. Thực tế đo từ dữ liệu có khoảng 20% đơn giao trễ, cao hơn nhiều so với SLA ở bảng trên.
- Chi phí vận hành là kịch bản giả định, không phải số liệu kế toán của XeDu.
- Mỗi đợt chỉ có 5 đến 16 đơn; các cỡ 14, 15, 16 mỗi cỡ chỉ có một đợt.

## Các nhánh khác

- `proposed-routing-rl`: Proposed RL và mô phỏng lập kế hoạch theo ngày.
- `feature/frontend-duy`: giao diện web và phân tích trên dữ liệu thật.
