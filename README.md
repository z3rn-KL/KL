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