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

## Lập kế hoạch theo ngày

Phần này trả lời hai câu hỏi rất thực tế. Thứ nhất, khi một shipper đã làm đủ 10 giờ mà vẫn còn đơn chưa giao thì nên cho làm thêm giờ hay dời số đơn đó sang hôm sau? Thứ hai, những cụm chỉ có vài đơn thì nên giữ một shipper riêng, hay giao luôn cho shipper ở cụm gần nhất?

Để so sánh, hệ thống mô phỏng 30 ngày liên tiếp (27/06 đến 26/07/2024, gồm 562 đơn thật). Mỗi shipper làm tối đa 10 giờ một ngày, mất 10 phút ở mỗi điểm giao, và xe chở được 100 kg. Đơn nào chưa giao kịp sẽ chuyển sang hôm sau và được ưu tiên giao trước.

Dữ liệu thật khá nhẹ so với giới hạn 10 giờ: trung bình mỗi ngày chỉ có khoảng 2,3 shipper hoạt động, và ngày đông nhất cũng chỉ 33 đơn. Vì vậy ở mức tải thật (x1) và gấp đôi (x2), không shipper nào làm quá ca. Để thấy rõ tác động, số đơn được nhân lên từ x4 đến x16 bằng cách lấy mẫu lại từ chính dữ liệu XeDu (giữ nguyên vị trí, cân nặng và hạn giao), và nhóm shipper được cố định ở 4 người. Các hệ số chi phí như lương giờ, hệ số làm thêm giờ 1,5 hay mức phạt dời đơn chỉ là giả định của mô phỏng, không phải chi phí thật của XeDu.

### Các cách xử lý được so sánh

Khi shipper hết ca mà còn đơn:

- **`overtime`**: cho làm thêm giờ đến khi giao hết, không giới hạn.
- **`defer`**: dừng đúng giờ, đơn còn lại chuyển sang hôm sau.
- **`hybrid`**: cho làm thêm tối đa 2 giờ, và chỉ giao tiếp khi tiền làm thêm giờ rẻ hơn mức phạt dời đơn.

Với các cụm nhỏ:

- **`dedicated`**: mỗi cụm một shipper riêng.
- **`merge_small`**: cụm dưới 3 đơn được giao cho shipper của cụm gần nhất, miễn là shipper đó vẫn xong việc trong ca.

### Kết quả

Chi phí trung bình mỗi đơn (VND), với cách phân bổ `dedicated`:

| Mức tải | `overtime` | `defer` | `hybrid` |
|---|---|---|---|
| x4 | 41,981 | 43,499, tồn 11 đơn | 42,210, tồn 1 đơn |
| x12 | 37,950 | 434,751, tồn 1,619 đơn | 183,105, tồn 898 đơn |

Một vài điều rút ra từ các con số này:

- **Dời đơn hoàn toàn là cách tệ nhất khi quá tải.** Đơn tồn cộng dồn qua từng ngày; ở x12, sau 30 ngày vẫn còn 1,619 đơn chưa giao và chi phí cao gấp hơn 11 lần so với làm thêm giờ.
- **Làm thêm giờ không giới hạn rẻ nhất, nhưng không thực tế.** Ở x12 có 53 ca shipper làm thêm quá 2 giờ. Nếu coi 2 giờ mỗi ngày là mức trần, chính sách tốt nhất là `hybrid`, rẻ hơn dời hẳn khoảng 2,4 lần. Vì vậy báo cáo chỉ chọn chính sách tốt nhất trong số những chính sách không vượt trần.
- **Khi tải vượt hẳn năng lực, thêm người hiệu quả hơn mọi chính sách.** Ở x12, tăng từ 4 lên 6 shipper đưa chi phí `hybrid` từ 183,105 xuống 57,502 VND mỗi đơn, và đơn tồn từ 898 xuống 129.
- **Gộp cụm nhỏ có lợi khi tải nhẹ.** Trên dữ liệu thật, `merge_small` giảm chi phí khoảng 12% (99,253 xuống 87,378 VND mỗi đơn) và chỉ cần 2,97 shipper thay vì 3,47. Cái giá là tỷ lệ giao đúng hạn giảm nhẹ, từ 100% xuống 99,5%. Khi tải cao, lợi ích này gần như mất đi vì số cụm đã bị giới hạn bởi số shipper.

### Hạn chế

- Mọi đơn được coi là phát sinh vào đầu ngày làm việc, và hạn giao là thời gian chờ ban đầu của đơn.
- `hybrid` chỉ cân nhắc mức phạt của một ngày dời tiếp theo, không tính đến việc hôm sau cũng có thể quá tải.
- Định tuyến trong mô phỏng dùng Nearest Neighbor kết hợp 2-opt, chưa dùng RL.

### Chạy lại

```bash
cd python
python experiments/daily_planning_prepare_matrix.py   # khoảng 46 giây, tạo ma trận đường đi
python experiments/daily_planning_experiment.py       # khoảng 2 phút
```

Kết quả nằm trong `results/`: `daily_planning_findings.txt` (bản tóm tắt), cùng ba file `daily_planning_summary.csv`, `daily_planning_daily.csv` và `daily_planning_sensitivity.csv`. Ma trận đường đi (`daily_planning_road_matrix.npz`, 25 MB) không được commit, nên cần chạy lệnh đầu tiên để dựng lại.

---

## Proposed RL

Agent học chọn điểm giao tiếp theo bằng cách chấm điểm từng đơn còn lại (một Q-network nhận 8 đặc trưng của mỗi ứng viên). Phần thưởng ở mỗi bước là phần tăng thêm của hàm mục tiêu J, lấy dấu âm; J kết hợp quãng đường, thời gian di chuyển, số đơn và số phút trễ, và số lần đổi cụm.

Các workload được chia thành 76 để huấn luyện, 16 để kiểm định (validation) và 16 để kiểm tra cuối (test). Hàm mục tiêu và checkpoint chỉ được chọn dựa trên train và validation, rồi khóa lại trước khi chạm vào test.

Kết quả trên tập test (16 workload, 131 đơn), đo bằng cùng hàm mục tiêu J đã khóa (càng thấp càng tốt):

| CW | DQN | Q-Learning | NN | SARSA | Proposed RL |
|---|---|---|---|---|---|
| 0.4257 | 0.4352 | 0.4495 | 0.4515 | 0.4575 | 0.4649 |

Nói thẳng: **Proposed RL không thắng các baseline trên tập test.** Trên validation, nó cũng chỉ ngang Nearest Neighbor (J = 0.4688 so với 0.4640). Đã thử thêm 5 đặc trưng "nhìn trước" (quãng đường về depot, độ vòng thêm, khoảng cách tới các điểm còn lại; xem `python/rl/proposed/lookahead_encoder.py`), nhưng kết quả còn tệ hơn trên validation (J trung bình 0.515 qua 3 seed), nên không đem sang đánh giá test. Chi tiết ở `results/proposed_rl_lookahead_validation_findings.txt`.

Một lời giải thích hợp lý là mỗi workload chỉ có từ 5 đến 16 điểm, nên các heuristic đã gần tối ưu và RL chưa có chỗ để cải thiện thêm. Đây là nhận định từ kết quả đo, chưa được kiểm chứng riêng.

---

## API cho các kết quả mới

| Endpoint | Tham số |
|---|---|
| `GET /api/results/daily-planning/summary` | `load_multiplier`, `allocation`, `overtime_policy` (đều tùy chọn) |
| `GET /api/results/daily-planning/daily` | `load_multiplier`, `allocation`, `overtime_policy` (bắt buộc cả ba) |
| `GET /api/results/daily-planning/sensitivity` | `parameter` (tùy chọn) |
| `GET /api/results/proposed-rl/final-test` | không có |

`allocation` nhận `dedicated` hoặc `merge_small`; `overtime_policy` nhận `overtime`, `defer` hoặc `hybrid`. Endpoint `summary` còn trả thêm `best_within_cap`: chính sách rẻ nhất ở từng mức tải trong số những chính sách không vượt trần làm thêm giờ.
