# 01 — Đọc luồng chạy, từ lệnh đến một batch

## A. Khi gõ `python train.py --config configs/demo.json`

1. `train.py` gọi `hagct.engine.trainer.main()` đọc CLI.
2. `train(args)` gọi `load_config()`. Những key viết sai bị báo lỗi; key bỏ trống dùng default trong `config.py`.
3. Đặt seed, chọn CPU/CUDA, đọc manifest và kiểm tra split/nhãn/file/signer.
4. Tạo `SkeletonDataset` cho train và val. Chưa tạo dataset test. Kiểm tra manifest có thể stat file test nhưng **không đọc tensor test trong vòng train**.
5. Tạo `FourStreamHAGCT`. Nếu có `--pretrained`, nạp encoder vào bốn stream trước khi tạo optimizer.
6. Tạo AdamW, scheduler warmup–cosine, GradScaler. Nếu resume, khôi phục toàn bộ trạng thái.
7. Mỗi epoch: đặt epoch augmentation → DataLoader train → `run_epoch(training=True)` → validation → lưu checkpoint → ghi log.

## B. `Dataset.__getitem__()` trả những gì?

| Bước | Nơi thực hiện | Kết quả |
|---|---|---|
| Đọc `.npy` | `dataset._read()` | Ví dụ raw `(T,27,3)` với XY-confidence |
| Bỏ padding cũ có khai báo | `preprocess()` | Chỉ lấy `length` frame đầu nếu manifest có cột length |
| Nội suy khớp mất + lọc | `preprocess()`, `filter.py` | Tọa độ XY hợp lệ; không lọc confidence |
| Chuẩn hóa | `preprocess()` | Root-relative, tùy chọn scale theo vai |
| Đổi trục, giảm clip quá dài | `preprocess()` | `(2,T,27)`, `T <= max_frames` |
| Augmentation | `augment_and_pad()`; chỉ train | Crop/speed/rotate/bone-scale/noise/joint-mask |
| Padding + mask đồng thời | `pad_sequence()` | `x=(2,Tmax,27)`, `mask=(Tmax,)` |
| Collate | PyTorch DataLoader | `x=(B,2,Tmax,27)`, `mask=(B,Tmax)`, `label=(B,)` |

Train có thể đặt clip vào giữa vùng padding; val/test luôn pad cuối. `True` trong mask = frame thật, không có nghĩa tất cả các joint của frame đó đều quan sát được. Tập hiện tại dùng **frame mask**, chưa có learned missing-joint mask.

## C. `FourStreamHAGCT.forward(x, mask)`

`build_streams()` sinh bốn tensor từ **cùng một mẫu đã augmentation**. Không augment độc lập bốn stream vì sẽ phá liên hệ hình học/thời gian.

Với mỗi stream, `encoder(signal, mask)` trả `(B,27,D)`, `head(features)` trả `(B,K)`. Bốn tensor logits được stack `(4,B,K)`, nhân softmax fusion weights `(4,1,1)`, cộng thành `(B,K)`.

Khi training, truyền **logits thô** vào CE/Focal; không softmax trước loss. Khi predict, softmax logits cuối rồi lấy top-k. Có thể gọi `return_streams=True` để xem logits của từng stream.

## D. Backward và checkpoint

`run_epoch()` lấy loss từng mẫu, cộng loss trong micro-batch, backward. Đến `accum_steps` hoặc batch cuối, hàm `optimizer_update()` unscale rồi chia gradient cho **tổng số mẫu thực tế** trước khi clip và step. Cách này không làm nhóm cuối ít mẫu bị sai trọng số.

Validation chạy `model.eval()` và không ghi gradient. Checkpoint tốt nhất dựa trên val top-1, không dựa trên test. Pretrain dùng val reconstruction MSE thấp nhất.

## E. Cách tự đọc code không bị rối

Đọc theo thứ tự: `configs/demo.json` → `trainer.train` → `SkeletonDataset.__getitem__` → `FourStreamHAGCT.forward` → `HAGCTEncoder.forward` → `run_epoch`. Chỉ mở `blocks.py` khi cần hiểu công thức trong encoder.

Để quan sát thay vì chỉ đọc, chạy `python -m scripts.trace_flow --config configs/demo.json`. Script dùng model ngẫu nhiên, chỉ minh họa shape và lời gọi; nó không train và không đánh giá chất lượng.
