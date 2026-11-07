# NEXUS Unified Clean — nguồn và phạm vi phát hành

Đây là bản đóng gói lại có chọn lọc từ `KLTN_NEXUS_FINAL.zip` do người dùng cung cấp. Toàn bộ tệp nguồn, thuật toán và kết quả nghiên cứu được **giữ nguyên byte nội dung**; công việc dọn chỉ loại các bản source UI cũ, ảnh kiểm thử, báo cáo UI trung gian và tập dữ liệu backup bị trùng hoàn toàn.

## Nguồn tích hợp

Theo báo cáo hợp nhất đính kèm từ Antigravity, nguồn đầu vào đã kết hợp: 
- Frontend Civic Atlas của người dùng từ `KLTN_ATLAS_REBUILD/KLTN_SHADCN`.
- Backend/thuật toán/kết quả từ hai nhánh GitHub `main` và `proposed-routing-rl` qua `KL_MERGED`.

Các commit nêu trong báo cáo: `00559c8` và `5426c89`. Đây là **thông tin theo báo cáo**, không phải lịch sử Git nằm trong ZIP. Bản này **không xác minh trực tuyến** những commit mới hơn được đẩy lên sau snapshot.

## Giữ nguyên

- `frontend/src/`, `frontend/public/`, cấu hình và package lock, bộ smoke/nav/page test chủ chốt.
- `python/` toàn bộ FastAPI, thuật toán, thí nghiệm và test. Bao gồm `python/rl/proposed/lookahead_encoder.py` và `python/simulation/daily_planning.py`.
- `dataset/`, `maps/` nguyên vẹn.
- `results/` nguyên vẹn ngoài ba tệp backup có nội dung *giống hệt* tệp chính; trọng số `.pt`, cache `.npz`, dữ liệu và kết quả nghiên cứu đều được giữ lại.
- `audit_outputs/` ba tệp kiểm toán khoa học độc lập.
- `docs/`, README, requirements, Docker/Compose.

## Loại khỏi gói sạch

- `frontend/src_checkpoint_*`: ba snapshot source UI dự phòng (145 tệp).
- `audit/`, `audit_outputs/V11_UI_REDESIGN/`, `frontend/audit/`: ảnh/nhật ký review UI không phục vụ runtime.
- `ATLAS_REBUILD_FINAL/`, `UI_REDUCTION_REVIEW/`: hồ sơ thiết kế/thử nghiệm trung gian.
- Script điều tra debug/capture tạm trong `frontend/tests/`, chỉ giữ smoke và audit chức năng có mục đích nghiệm thu.
- `results/benchmark_backup_2026_10_06/`: 3 tệp CSV đã được đối chiếu và giống từng byte với 3 tệp ở `results/`.
- File prompt/báo cáo UI/V12 cũ nằm ở gốc, cấu hình `.vscode` cá nhân.

## Điểm cần xác minh trước khi gọi là release học thuật cuối

1. `frontend/src/pages/RoutesPage.tsx` đang lấy `reference_metrics`, `evaluation`, `route_geometry` của Guided DQN để ghép với thứ tự giao hàng Vanilla DQN. Đồng thời overlay đối chứng DQN có lời gọi `nearest_neighbor` trên danh sách ID được sắp theo thứ tự Vanilla DQN. Điều này **có thể làm sai số liệu/hình học hiển thị**. Bản đóng gói sạch **không sửa nghiệp vụ**; cần điều chỉnh nguồn API đúng theo thuật toán và kiểm thử đối chiếu thực nghiệm trước khi dùng số liệu này để bảo vệ khóa luận.
2. Không có hai tệp `results/daily_planning_road_matrix.npz` và `results/proposed_rl_lookahead_D_best.pt`. Tệp thứ nhất là đầu vào cần để tái chạy thí nghiệm Daily Planning; tệp thứ hai là đầu ra mô hình sau khi train Lookahead. Không suy diễn rằng mọi thực nghiệm có thể tái lập từ ZIP mà không cần bước chuẩn bị/tái huấn luyện.
3. Phần giải thích của báo cáo gốc về tuyến 108 WL là giới hạn lưu trữ dữ liệu *delivery_order*, không tự chứng minh thuật toán sai. Không tự bịa tuyến hoặc số liệu.
4. Giữ 9 URL để tương thích. Đừng nhầm việc đóng gói sạch với việc đã kiểm thử lại API/build/Playwright end-to-end.

## Kiểm tra toàn vẹn

`RELEASE_SHA256.csv` liệt kê mã băm SHA-256 của **mỗi file được giữ nguyên từ bản đầu vào** và các tệp tài liệu bổ sung; `scripts/verify_release.py` đối chiếu mã băm sau khi giải nén. Manifest không tự bao gồm chính nó.
