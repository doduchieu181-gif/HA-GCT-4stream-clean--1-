# 05 — Đối chiếu với nhánh tham khảo và hướng cải tiến

Nguồn đối chiếu: repository `KILIAN0802/HA-GCT`, nhánh **`hagct`**, snapshot commit `693a2788492f5a0335a50cf50addadd1a2bf90f4`. Không đồng nhất nhánh này với `main` hoặc `ms`.

## 1. Giữ hướng nghiên cứu

| Ý tưởng | Trong bản clean | Nơi đọc |
|---|---|---|
| Bốn stream Joint/Bone/Motion/Bone-Motion | Có, bốn encoder/head độc lập | `models/four_stream.py` |
| Đồ thị chú trọng bàn tay | Có, body + hand topology và learned edge strength/gate | `models/blocks.py`, `topology.py` |
| Học phụ thuộc thời gian | Có, local temporal conv + temporal MHSA | `models/encoder.py` |
| Hợp nhất spatial/temporal | Có, **gated fusion** | `HAGCTEncoder.forward()` |
| Hợp nhất bốn stream | Có, softmax bốn tham số rồi weighted sum logits | `FourStreamHAGCT.forward()` |
| One Euro + xử lý khớp thiếu | Có, công thức lọc sửa lại + linear interpolation | `data/filter.py`, `preprocess.py` |
| Augmentation chuyển động/xương | Có, trên phần clip thật trước padding | `data/augment.py` |
| Masked reconstruction pretrain | Có, pooled encoder + MLP decoder đơn giản | `models/pretrain.py` |
| AMP/accumulation/focal/resume | Có, gói trong một trainer dùng chung | `engine/` |

## 2. Những vấn đề xử lý lại

| Vấn đề ở đường code tham khảo | Cách triển khai lại |
|---|---|
| Luồng/file và nhánh thử nghiệm khó lần theo | Một package `hagct`, bốn CLI mỏng, JSON config, docs theo luồng, `trace_flow` |
| One Euro có biểu thức alpha không đúng dạng RC low-pass | `alpha=dt/(dt+1/(2π cutoff))`; test số và step signal |
| Noise/augmentation đụng frame padding hoặc mask không đi cùng placement | Augment clip chưa pad; tạo tensor padded và mask trong cùng hàm |
| Sai phân ở biên padding có thể tạo motion giả | Motion chỉ có giá trị khi cả frame hiện tại/lần trước đều hợp lệ |
| Bone scaling dùng parent đã biến đổi để tính bone gốc | Tính toàn bộ bone từ input bất biến, rồi dựng lại theo parent đã scale |
| Temporal graph bias ràng buộc T=27 | Không ép graph V×V vào attention T×T; graph dành cho spatial, MHSA dành cho temporal |
| Residual FFN có nguy cơ được cộng hai lần | FFN thuần transform, block sở hữu đúng một residual |
| BatchNorm có thể trộn thống kê padding | LayerNorm theo feature + mask sau từng block |
| Tên stream/nạp pretrained không thống nhất | `joint,bone,motion,bone_motion` tập trung; strict load cả bốn encoder |
| Nhầm channel confidence với Z | Khai báo `xy`, `xy_conf`, `xyz`; không tự đoán |
| Pretrain gộp train+val làm mờ protocol | Update chỉ train; validation tách riêng; test ngoài trainer |
| Tích lũy gradient nhóm cuối ngắn | Chuẩn hóa gradient theo tổng mẫu thực của từng update |
| Rủi ro ghi đè thí nghiệm | Run mới từ chối output không rỗng; resume kiểm tra cấu hình/manifest |

Các sửa trên đã có kiểm thử tương ứng ở `tests/`; chúng **không thay thế benchmark accuracy**.

## 3. Thay đổi có chủ đích, KHÔNG phải refactor tương đương số học

- Dùng LayerNorm, depthwise local conv và projection thống nhất thay các chi tiết BatchNorm/Conv trong implementation tham khảo.
- Adaptive graph ở đây là **positive normalized edge reweighting trên support cố định**, không tái tạo mọi biến thể HA-GC/dynamic adjacency của repo/paper.
- Dùng một D cố định; số spatial/temporal layer configurable. Không mang theo toàn bộ tùy chọn block cũ.
- Normalize root-relative + median shoulder scale, resize toàn clip dài, khai báo format rõ ràng. Các lựa chọn này có thể làm thay đổi phân phối dữ liệu so với run cũ.
- Pretrain masked-token có positional embedding, decoder MLP từ feature đã temporal-pool; không cam kết giống objective/decoder của công trình gốc.
- Frame mask xuyên suốt nhưng chưa dùng per-joint visibility mask trong graph/pooling/loss. Điểm nội suy và điểm mất toàn clip vẫn là hạn chế dữ liệu.

Do đó **không load thẳng checkpoint cũ**, không so accuracy cũ/mới nếu chưa thống nhất preprocessing, split và training budget. Gọi bản này là “clean baseline theo hướng HA-GCT 4-stream”, không tuyên bố tái lập hoàn toàn paper.

## 4. Những ý tưởng chưa đưa vào luồng chính

| Ý tưởng | Trạng thái | Muốn bổ sung cần làm gì? |
|---|---|---|
| Bidirectional Cross-Attention S↔T | Chưa triển khai | Thiết kế token S/T, attention masks, ablation với gated fusion |
| STGAIN / graph imputation học được | Chưa triển khai | Missingness mask/targets, train objective, kiểm thử leakage và reconstruction |
| Conditional skeleton diffusion | Chưa triển khai | Train/sampling nhất quán, chỉ sinh từ train, đánh giá chất lượng và hiệu quả phân loại |
| Video/RGB → 27 joint | Chưa triển khai | Extractor + mapping topology + FPS/confidence contract |
| DDP / multi-GPU / export ONNX / TTA | Chưa triển khai | Tối ưu theo môi trường sau khi baseline đúng |
| Checkpoint migration từ repo cũ | Chưa triển khai | Mapping có kiểm tra từng key và chứng minh semantic compatibility |

Không có class rỗng hoặc flag giả tạo cảm giác các ý tưởng này đã hoạt động. Nếu triển khai tiếp, thêm module/config/test riêng rồi tích hợp qua interface đã có.
