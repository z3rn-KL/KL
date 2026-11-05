# Tối ưu tuyến giao hàng kết hợp phân cụm và học tăng cường (KLTN_11 — Unified)

Mã nguồn khóa luận: **XÂY DỰNG HỆ THỐNG TỐI ƯU TUYẾN GIAO HÀNG KẾT HỢP KỸ THUẬT PHÂN CỤM VÀ REINFORCEMENT LEARNING TRONG LOGISTICS**.  
Hệ thống kết hợp phân cụm không gian (K-Means, DBSCAN), các thuật toán heuristic kinh điển (Nearest Neighbor, Clarke-Wright), các thuật toán học tăng cường định tuyến (Q-Learning, SARSA, Vanilla DQN, K-Means-guided DQN, Proposed Candidate-Scoring DQN) và học tăng cường cấp ca làm việc (Shift-level RL: DQN, Policy Gradient, Set Attention Network).

Toàn bộ mã nguồn và kết quả thực nghiệm từ cả ba nhánh GitHub (`main`, `proposed-routing-rl`, `feature/frontend-duy`) đã được hợp nhất vào một cây làm việc duy nhất, chuẩn xác và vượt qua 464/464 bài kiểm thử tự động.

---

## 1. Dữ liệu và Hướng dẫn thực thi

Dữ liệu và đồ thị đường **không nằm trong repo** (đã cấu hình trong `.gitignore` theo quy định bảo vệ dữ liệu PII và giới hạn dung lượng Git):
- Đặt tệp đơn hàng chuẩn tại: `dataset/xedu/xedu_cleaned.csv` (2.394 đơn).
- Đặt đồ thị giao thông tại: `maps/xedu_drive.graphml` (166 MB trích xuất từ OpenStreetMap TP.HCM).
- Toàn bộ kết quả thực nghiệm, ma trận đường và checkpoint mô hình đã tính sẵn trong `results/`.

Yêu cầu môi trường: **Python 3.12+** và **Node.js 18+**. Bộ test: **464 tests**.

```bash
# Cài đặt thư viện Python
pip install -r requirements.txt

# Chạy toàn bộ 464 bài kiểm thử hồi quy
pytest python
```

**Trên Windows PowerShell** (từ thư mục gốc; thư mục `.venv` tạo trên Linux không dùng lại được, cần xóa và tạo mới):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pytest python
```

### Khởi động Backend API (Cổng 8000)

**Trên Linux / macOS / Bash:**
```bash
cd python && PYTHONPATH=. uvicorn api.main:app --port 8000
```

**Trên Windows Command Prompt (CMD):**
```cmd
cd python
set PYTHONPATH=.
uvicorn api.main:app --port 8000
```

**Trên Windows PowerShell:**
```powershell
cd python
$env:PYTHONPATH="."
$env:PYTHONUTF8="1"
..\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
```

### Khởi động Giao diện Web NEXUS (Cổng 5173 / 5174)

```bash
cd frontend
npm install
npm run dev
```

Build giao diện phục vụ sản xuất: `cd frontend && npm run build`. Chi tiết các trang và API endpoints: [frontend/README.md](frontend/README.md).

---

## 2. Quyền riêng tư và Cảnh báo an toàn trước khi triển khai

Dữ liệu là đơn hàng thật từ XeDu Logistics TP.HCM. Endpoint `GET /api/deliveries` **mặc định đã được ẩn danh hóa**:
- Loại bỏ mã nhân viên giao hàng (`shipper`, `shipper_id`) và đường dẫn ảnh biên nhận.
- Cắt gọn địa chỉ người gửi và người nhận còn đến cấp quận/huyện.
- Làm tròn tọa độ địa lý 3 chữ số thập phân (~110 m sai số không gian).
- Quá trình ẩn danh thực hiện trước khi tìm kiếm, do đó tham số `q` không thể dò ra số nhà thực tế.
- Chỉ khi phát triển trên máy cá nhân mới được tắt ẩn danh bằng biến môi trường `XEDU_ANONYMIZE=0`; **tuyệt đối không bao giờ tắt trên máy chủ công khai**. Mã đơn (`id`, `mdh`) và tên gói hàng được bảo lưu để giao diện vận hành.

**Lưu ý triển khai công khai:**
- Đặt biến `APP_PASSWORD` sẽ bật **chế độ công khai** (`python/api/public_mode.py`): mọi đường dẫn đều cần mật khẩu, tọa độ trong phản hồi API được làm tròn còn khoảng 110 m (kể cả hình học tuyến đường), giao diện đã build được phục vụ cùng địa chỉ với API, và `/docs` bị ẩn. Không đặt biến này thì chạy như bình thường.
- Chạy công khai từ máy Windows (máy phải bật trong lúc có người truy cập; server cần khoảng 2,6 GB RAM):
  ```powershell
  cd frontend; npm ci; npm run build; cd ..
  winget install --id Cloudflare.cloudflared
  $env:APP_PASSWORD = "mat-khau-cua-ban"
  .\scripts\start_public.ps1 -Tunnel
  ```
  Địa chỉ `https://...trycloudflare.com` hiện trong terminal và đổi sau mỗi lần chạy lại.
- Các endpoint `/api/routing` và `/api/evaluation` chưa có cơ chế rate limiting và bị giới hạn cứng tối đa 30 đơn hàng mỗi lượt yêu cầu để tránh quá tải tài nguyên đồ thị bộ nhớ.

---

## 3. Phân tích thực nghiệm trên dữ liệu thực tế (Real-Data Replay)

Đầu vào là dữ liệu thật: danh sách đơn, tọa độ địa lý, hạn cam kết giao hàng (SLA), thời điểm giao thực tế của Shipper, và đồ thị đường phố OpenStreetMap. Quãng đường và thời gian của các tuyến đường được tính toán trên mô hình đồ thị với tốc độ lưu thông lý tưởng, **không phải số liệu đo đạc thực địa**, và chưa có thuật toán nào được triển khai vận hành ngoài đời thực.

### Sai lệch giữa mô phỏng và thực tế:
- **Thực tế quan sát giao trễ 20,3%** (trên 2.281 đơn hàng có thời hạn giao).
- **Mô hình thời gian lý tưởng lạc quan hơn thực tế rất nhiều:** Khoảng thời gian giữa hai lần giao liên tiếp trong thực tế có trung vị là **28,9 phút**, trong khi mô hình đường phố chỉ tính ra **6,9 phút** (chênh lệch trung vị **20,3 phút mỗi điểm dừng** dành cho việc tìm nhà, dừng đỗ, phục vụ khách hàng, gọi điện, nhận tiền và ùn tắc giao thông). Do đó, **tỷ lệ đúng hạn SLA 96% đến 100% trong các bảng benchmark là số liệu tính theo mô hình lý tưởng, không phản ánh tỷ lệ giao đúng hạn trong đời thực**.

### Phát lại 64 ca giao sạch (Clean Shifts):
Lọc 64 ca giao đơn lẻ (1 shipper, 1 điểm kho/lấy hàng, các đơn tạo trước khi giao; quy mô 3 đến 7 đơn, trung vị 3 đơn) và đối chiếu với nghiệm tối ưu chính xác (Exact Solver):

| Phương pháp | Dài hơn tối ưu (Thời gian) | Dài hơn tối ưu (Quãng đường) |
|---|---|---|
| **Thứ tự giao thực tế của Shipper** | **+10,8%** (CI 95%: 7,5% đến 14,5%) | **+11,6%** |
| **Nearest Neighbor** | +3,7% | +4,6% |
| **Nearest Neighbor + Local Search** | **+0,16%** | **+0,10%** |

Thực tế thứ tự giao của shipper không bám theo hạn giao (chỉ 16/64 ca tuân thủ đúng thứ tự hạn giao sớm nhất). Nghiệm tối ưu là cận dưới cho bài toán tĩnh nên khoảng cách này không phải con số có thể cam kết tiết kiệm 100% ngoài đời. Chạy lại thực nghiệm: `python experiments/real_data_replay.py`.

---

## 4. Các thành phần nghiên cứu và Kết quả đối chứng

### A. Phân cụm không gian & Tiền xử lý
- **K-Means ($K = 10$):** Tối ưu hóa điểm uốn (Elbow method) và hệ số Silhouette trên 2.394 đơn giao hàng tại TP.HCM.
- **DBSCAN ($\epsilon = 2.0\text{ km}, \text{min\_samples} = 8$):** Phân cụm theo mật độ không gian và phân tách các điểm giao ngoại lai, bất lợi.

### B. Định tuyến Heuristic & RL cơ sở
- **Nearest Neighbor (NN) & Clarke-Wright (CW):** Định tuyến dựa trên ma trận cự ly và thời gian của mạng đường phố OpenStreetMap.
- **Tabular Q-Learning & SARSA:** Tối ưu hóa thứ tự ghé thăm trên không gian trạng thái rời rạc.
- **Vanilla DQN & K-Means-guided DQN:** Mạng Q sâu kết hợp hướng dẫn cụm, giảm thiểu việc chuyển đổi cụm tùy tiện.

### C. Proposed Candidate-Scoring Routing RL
Q-network nhỏ (8 đặc trưng chuẩn hóa của ứng viên, 2 lớp ẩn 64 units), huấn luyện theo cơ chế DQN chấm điểm ứng viên để chọn điểm giao tiếp theo. Phần thưởng mỗi bước là trừ phần tăng thêm của hàm mục tiêu tổng hợp $J$ (gồm cự ly, thời gian, số đơn trễ, phút trễ SLA và số lần đổi cụm). Mã nguồn tại `python/rl/proposed/`.

Quy trình: 76 workload huấn luyện, 16 kiểm định, 16 kiểm tra độc lập. Hàm mục tiêu và checkpoint được khóa hoàn toàn trước khi đánh giá trên tập kiểm tra.

Kết quả trên tập kiểm tra độc lập (16 workload, 131 đơn hàng, chỉ số $J$ càng thấp càng tốt):

| Clarke-Wright | Guided DQN | Q-Learning | Nearest Neighbor | SARSA | Proposed RL |
|---|---|---|---|---|---|
| **0,4257** | 0,4352 | 0,4495 | 0,4515 | 0,4575 | **0,4649** |

> **Nhận định khoa học khách quan:** Proposed RL **không thắng các thuật toán đối chứng truyền thống (CW, NN)** trên bài toán định tuyến đơn lẻ này. Trên tập kiểm định, nó chỉ đạt ngang Nearest Neighbor ($J = 0,4688$ so với $0,4640$). Vì các workload trong dữ liệu XeDu có quy mô nhỏ (5 đến 16 điểm dừng), heuristic tổ hợp đã tiệm cận tối ưu toàn cục. Thử nghiệm mở rộng thêm 5 đặc trưng nhìn trước (Lookahead Encoder) cho kết quả kém hơn ($J$ trung bình 0,515 trên 3 seed) nên không đưa sang kiểm tra chính thức (`results/proposed_rl_lookahead_validation_findings.txt`).

### D. RL theo ca làm việc (Shift-Level RL)
Mô phỏng trọn vẹn một ca làm việc của shipper (8 đến 40 đơn hàng). Mỗi bước, agent chọn giao một đơn tiếp theo hoặc kích hoạt hành động **dừng ca** (các đơn còn lại chuyển giao sang ngày hôm sau).
- **Thưởng (Reward):** Hoa hồng 3% giá trị đơn nếu đúng hẹn (tối đa 10% nếu giao sớm), thưởng kết thúc ca sớm.
- **Phạt (Penalty):** Cự ly, thời gian, làm thêm giờ, đơn trễ, biên lợi nhuận thấp dưới 30%, đơn dời, chi phí khách chờ đợi theo phút (`customer_waiting_cost`).
- **Mã nguồn:** `python/rl/shift/` (gồm DQN, Policy Gradient và Set Attention Network).

Kết quả trên tập test độc lập (46 ca làm việc từ 10 đến 50 đơn hàng), khoảng cách tới nghiệm tham chiếu tối ưu (Gap to Reference, càng thấp càng tốt):

| Phương pháp | Tất cả 46 ca (%) | Ca có nghiệm chính xác, 30 ca (%) | Thời gian suy luận / ca |
|---|---|---|---|
| **Nearest Neighbor + Local Search** | **2,94%** | **2,71%** | 0,21 giây |
| **Set Attention Network + Behavioral Cloning** | **6,60% ± 0,95** | **5,65% ± 0,20** | **5,6 ms** |
| Set Attention Network (học từ đầu) | 10,70% ± 0,05 | 10,74% ± 0,33 | 5,6 ms |
| Policy Gradient MLP + Behavioral Cloning | 8,12% ± 0,40 | 7,45% ± 0,25 | 3,2 ms |
| Shift DQN (hai biến thể) | 59% – 62% | 47% – 52% | 4,1 ms |

DQN thất bại ở cấp độ ca làm; Policy Gradient và Attention Network học thành công. Mô hình Set Attention suy luận chỉ mất 5,6 ms (nhanh gấp 37 lần so với Local Search 210 ms).

### E. Lập kế hoạch theo ngày (Daily Planning Simulation)
Mô phỏng 30 ngày liên tục (562 đơn thật), ca 10 giờ, 10 phút phục vụ mỗi điểm. Đơn chưa giao chuyển sang ngày hôm sau. So sánh ba chính sách xử lý cuối ca (`overtime`, `defer`, `hybrid` làm thêm tối đa 2 giờ) và hai cách phân bổ cụm (`dedicated`, `merge_small`). Mã nguồn: `python/simulation/daily_planning.py`.

Ở mức tải cao x12 (kịch bản mô phỏng từ dữ liệu thật):
- Chính sách `overtime`: Chi phí 37.950 VND/đơn nhưng có tới 53 ca vi phạm trần làm thêm 2 giờ.
- Chính sách `defer`: Chi phí 434.751 VND/đơn, tồn đọng 1.619 đơn hàng do tắc nghẽn tích lũy.
- Chính sách `hybrid`: Chi phí 183.105 VND/đơn, tồn 898 đơn — **chính sách tốt nhất thỏa mãn trần làm thêm giờ**.
- Gộp cụm ít đơn (`merge_small`) giúp giảm khoảng 12% chi phí ở mức tải thực tế (99.253 xuống 87.378 VND mỗi đơn).

---

## 5. Giới hạn nghiên cứu & Giả định mô hình

1. **Hệ số tiền tệ là giả định:** Các hệ số tiền tệ (lương shipper 35.000 VND/h, làm thêm giờ 52.500 VND/h, hoa hồng, phạt trễ 10.000 VND, phạt dời đơn 50.000 VND) là các tham số giả định kinh tế trong mô hình; tập dữ liệu XeDu không chứa giá trị thanh toán của từng đơn hàng.
2. **Kịch bản sản lượng mô phỏng:** Dữ liệu thực tế XeDu phân bổ trong ngày không bao giờ vượt ca 10 giờ; do đó các mức tải x4 đến x16 trong Daily Planning là các kịch bản nhân tải mô phỏng để đánh giá ngưỡng chịu tải của thuật toán, không phải dữ liệu thực tế phát sinh.
3. **Giới hạn thời gian phục vụ tĩnh:** Mô phỏng mặc định thời gian phục vụ tại mỗi điểm là 10 phút cố định; thực tế thời gian này biến thiên phụ thuộc vào việc tìm khách, gửi xe và liên lạc.

---

## 6. Hồ sơ kiểm toán và Tài liệu nghiệm thu

Toàn bộ báo cáo kiểm toán khoa học và hồ sơ kỹ thuật được lưu trữ trong thư mục `docs/`:
1. [KLTN_11_THREE_BRANCH_GIT_AUDIT.md](docs/KLTN_11_THREE_BRANCH_GIT_AUDIT.md): Kiểm toán pháp y ba nhánh Git.
2. [KLTN_11_UNIFIED_SOURCE_INVENTORY.md](docs/KLTN_11_UNIFIED_SOURCE_INVENTORY.md): Bảng kiểm kê chi tiết toàn bộ các module mã nguồn.
3. [KLTN_11_UNIFIED_DATASET_CHECKPOINT_AUDIT.md](docs/KLTN_11_UNIFIED_DATASET_CHECKPOINT_AUDIT.md): Kiểm toán băm SHA-256 dữ liệu và 18 checkpoints.
4. [KLTN_11_UNIFIED_RL_PROVENANCE.md](docs/KLTN_11_UNIFIED_RL_PROVENANCE.md): Nguồn gốc khoa học và phân định Proposed RL vs Shift RL.
5. [KLTN_11_THREE_BRANCH_INTEGRATION_MATRIX.md](docs/KLTN_11_THREE_BRANCH_INTEGRATION_MATRIX.md): Ma trận quyết định tích hợp liên nhánh.
6. [KLTN_11_UNIFIED_BACKEND_API_MAPPING.md](docs/KLTN_11_UNIFIED_BACKEND_API_MAPPING.md): Ánh xạ 26 endpoints API và phân loại bản chất hoạt động.
7. [KLTN_11_UNIFIED_SYSTEM_ACCEPTANCE.md](docs/KLTN_11_UNIFIED_SYSTEM_ACCEPTANCE.md): Báo cáo nghiệm thu hệ thống toàn diện.
8. [NEXUS_UNIFIED_WEBSITE_FOUNDATION_PLAN.md](docs/NEXUS_UNIFIED_WEBSITE_FOUNDATION_PLAN.md): Bản thiết kế và lộ trình phát triển Website NEXUS Delivery Intelligence.
