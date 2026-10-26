# BẢNG KIỂM THỬ THỦ CÔNG HỆ THỐNG (MANUAL QA CHECKLIST)
Hệ thống Tối ưu Tuyến giao hàng Logistics XeDu (TP.HCM) — KLTN_11

Tài liệu hướng dẫn kiểm thử thủ công giao diện và tương tác người dùng trên các màn hình, chế độ hiển thị và điều kiện chuyển động.

---

## 1. Môi trường & Thiết bị Kiểm thử
- **Độ phân giải hiển thị:**
  - Desktop lớn: `1920px × 1080px`
  - Laptop tiêu chuẩn: `1440px × 900px`
  - Tablet ngang / Màn hình nhỏ: `1024px × 768px`
  - Tablet dọc: `768px × 1024px`
  - Mobile: `390px × 844px` (iPhone 12/13/14 Pro)
- **Chế độ giao diện (Themes):** Dark Mode (mặc định nền graphite) & Light Mode.
- **Tùy chọn chuyển động:** Bình thường & `prefers-reduced-motion: reduce`.

---

## 2. Danh mục Kiểm thử Theo Từng Trang

### 2.1. Trang Tổng quan (`/tong-quan`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Hero bản đồ nền | Điểm cụm phân bố đúng 10 màu, tuyến vẽ dần mượt mà, không tô khối nền lớn | [ ] |
| Tóm tắt khoa học | Hiển thị cam kết SLA 100%, tỷ lệ đổi cụm (-6,83%), mức đánh đổi quãng đường (+1,76%) / thời gian (+1,87%) lấy từ API | [ ] |
| Stepper 5 bước | Nối bằng đường đứt nét, tự đổi trạng thái hoàn thành/kích hoạt theo vị trí cuộn, hiển thị dạng dọc trên màn hình nhỏ (<768px) | [ ] |
| Thẻ chỉ số KPI | Vòng gauge SLA, thanh phân kỳ chênh lệch %, số lớn (2.394 đơn, 16 shipper) | [ ] |
| Ghi chú trung thực | Callout chi phí là mô phỏng, so sánh cùng điều kiện dữ liệu | [ ] |
| Chuyển đổi Dark/Light | Màu chữ, thẻ card và nền chuyển đổi đồng bộ, độ tương phản cao | [ ] |

### 2.2. Trang Quản lý Đơn hàng (`/don-hang`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Khung mục tiêu nghiên cứu | Khung ở đầu trang thể hiện rõ mục tiêu 2.394 đơn XeDu | [ ] |
| Vạch màu nhận diện cụm | Mép trái mỗi dòng bảng có vạch màu tương ứng với cụm K-Means (0–9) | [ ] |
| Chú thích thời hạn SLA | Chú thích rõ ràng ký hiệu `–` = không có hạn giao (SLA linh hoạt) | [ ] |
| Panel chi tiết đơn hàng | Khi bấm vào dòng, panel trượt vào từ phải; có nút `✕` đóng panel; hiển thị bản đồ vị trí điểm giao | [ ] |
| Bộ lọc và tìm kiếm | Lọc theo dịch vụ, kết quả đúng hạn/trễ, cụm, sắp xếp tăng/giảm hoạt động trơn tru | [ ] |
| Kết luận động cuối trang | Số liệu tổng đơn, shipper, đơn không SLA tính động từ dữ liệu API | [ ] |

### 2.3. Trang Phân cụm Địa lý & Sàng lọc Điểm bất lợi (`/phan-cum`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Bản đồ không gian lớn | Chiều cao bản đồ đạt 700px, hiển thị đầy đủ 2.394 điểm giao và tâm cụm | [ ] |
| Khung mục tiêu nghiên cứu | Hiển thị ở đầu trang với thông số K=10 và lọc DBSCAN | [ ] |
| Biểu tượng thuật ngữ (i) | Biểu tượng `(i)` cạnh `Silhouette`, `eps`, `min_samples` hiển thị tooltip tiếng Việt khi hover | [ ] |
| Chuyển đổi thuật ngữ/chế độ | Chuyển giữa K-Means chuẩn, K-Means tùy chỉnh và DBSCAN hiển thị đúng tham số | [ ] |
| Bảng điểm bất lợi xa tâm | Nhấp vào dòng điểm bất lợi sẽ tự động zoom cận cảnh địa chỉ thực tế trên bản đồ OSM | [ ] |
| Khung kết luận chính | Thống kê số lượng cụm, chênh lệch min/max và số điểm xa tâm tính động | [ ] |

### 2.4. Trang Tuyến giao hàng (`/tuyen-giao`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Tuyến đường động | Nét tuyến tương phản cao (cyan `#00f0ff`), hiệu ứng dash animation theo chiều đi | [ ] |
| Kho và điểm giao | Kho hàng (Depot) nhấp nháy nhẹ; điểm giao xuất hiện lần lượt theo thứ tự phát; số thứ tự rõ nét | [ ] |
| Nhãn Mạng đường thực OSM | Hiển thị nhãn "Mạng đường thực OSM" trên HUD; nếu có đoạn ước lượng vẽ nét đứt và ghi chú rõ | [ ] |
| Chú giải bản đồ | Chú thích đường nét liền = đồ thị OSM, nét đứt = đoạn ước lượng/so sánh trực tiếp | [ ] |
| Phím tắt Space | Nhấn phím Space để tạm dừng / tiếp tục animation phát tuyến | [ ] |
| So sánh NN & Clarke–Wright | Nút "Chạy so sánh NN & Clarke–Wright" tính toán và vẽ tuyến song song chính xác | [ ] |

### 2.5. Phòng Thử nghiệm Thuật toán (`/chay-thu`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Chọn nguồn dữ liệu | Chọn theo Cụm K-Means, Workload benchmark hoặc chọn thủ công đơn hàng | [ ] |
| Chạy thử nghiệm trực tiếp | Bấm nút chạy, backend tính toán lộ trình NN và Clarke–Wright trực tiếp trên đồ thị đường | [ ] |
| Nhãn Mạng đường thực OSM | Hiển thị nhãn "Mạng đường thực OSM" trên thanh điều khiển bản đồ | [ ] |
| Bảng so sánh kết quả | So sánh quãng đường, thời gian, SLA, chi phí và thời gian tính toán | [ ] |
| Thứ tự ghé thăm | Danh sách đơn hàng được đánh số theo đúng trình tự xuất phát từ kho | [ ] |

### 2.6. Trang So sánh Phương pháp (`/so-sanh`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Bảng 4 phương pháp | Bảng so sánh chuẩn không bị tràn ngang, thanh cuộn mượt mà có gợi ý | [ ] |
| Công tắc chuyển đổi | Các công tắc hiển thị đủ nhãn, không bị cắt chữ ở mọi kích thước màn hình | [ ] |
| Biểu đồ chênh lệch % | Định dạng `0,00%` (không bị âm `-0,00%`), biểu đồ phân kỳ trực quan | [ ] |
| Chú thích khoa học | Chú thích rõ: "Guided DQN kém Vanilla DQN và Clarke–Wright về quãng đường và thời gian; đóng góp của nó là tính liên tục theo cụm" | [ ] |
| Biểu đồ thời gian huấn luyện | Biểu đồ dùng một tông màu thống nhất; cột nguồn ghi rõ tiếng Việt ("Huấn luyện mới" / "Dùng lại từ độ nhạy") | [ ] |

### 2.7. Trang K-Means-guided DQN (`/guided-dqn`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Công thức thưởng mở rộng | Khối công thức $R = R_{env} - \lambda \cdot P_{switch}$ rõ ràng, có tô sáng trọng số | [ ] |
| Sơ đồ tích hợp 4 khối | Trực quan hóa luồng dữ liệu từ K-Means sang Không gian trạng thái, Hàm thưởng và DQN Agent | [ ] |
| Khảo sát chuỗi cụm | Dòng "Δ quãng đường (G − V): +x% (Guided dài hơn)" tính chính xác từ dữ liệu thực nghiệm | [ ] |

### 2.8. Trang Độ nhạy & Kiểm chứng (`/do-nhay`) và Độ ổn định (`/on-dinh`)
| Hạng mục kiểm tra | Tiêu chí đạt (Pass Criteria) | Trạng thái |
| :--- | :--- | :---: |
| Thanh trượt $\lambda$ | Di chuyển thanh trượt qua các mốc 0, 0.10, 0.25, 0.50 cập nhật thẻ phân tích tương ứng | [ ] |
| Bảng độ nhạy 12 workload | Hiển thị đầy đủ số liệu cho cả 12 ca giao hàng thực tế | [ ] |
| Đánh giá hạt giống ngẫu nhiên | Bảng hạt giống và kiểm chứng tham số tải đầy đủ dữ liệu từ endpoint `/api/results/stability` | [ ] |

---

## 3. Kiểm thử Tương thích Giao diện & Hiệu năng

### 3.1. Đáp ứng Kích thước Màn hình (Responsive)
- [ ] **1920px & 1440px:** Bố cục dạng 2 cột rộng rãi, bản đồ và bảng dữ liệu hiển thị song song không che khuất.
- [ ] **1024px & 768px:** Menu chuyển sang co gọn hoặc thu hẹp thanh bên; các bảng dữ liệu tự động cho phép cuộn ngang (`overflow-x: auto`).
- [ ] **390px (Mobile):** Các khối xếp chồng theo chiều dọc (single column); các nút bấm và thanh trượt dễ thao tác bằng ngón tay.

### 3.2. Chế độ Chuyển động Tối thiểu (`prefers-reduced-motion`)
- [ ] Khi bật `prefers-reduced-motion: reduce`, tắt toàn bộ animation cuộn, không giật màn hình.
- [ ] Tuyến đường và bản đồ vẫn hiển thị tĩnh đầy đủ dữ liệu trong vòng $\le 800\text{ ms}$.

### 3.3. Kiểm tra Lỗi Console (Developer Tools)
- [ ] Mở Console (F12) duyệt qua toàn bộ 9 trang: **0 cảnh báo nghiêm trọng, 0 lỗi JavaScript/React**.

