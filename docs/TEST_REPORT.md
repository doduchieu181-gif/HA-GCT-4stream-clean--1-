# Báo cáo kiểm thử bản bàn giao

Ngày chạy: 2026-09-24. Phần lớn kiểm thử dùng dữ liệu tổng hợp; ngoài ra pipeline trial đã được chạy trực tiếp với `27kpt.zip`. Đây vẫn không phải benchmark nghiên cứu.

## Môi trường thực chạy

- Python 3.12.14
- PyTorch 2.6.0+cpu
- NumPy 2.3.5
- CPU, không có CUDA khả dụng
- Unit/integration tests đặt PyTorch CPU threads=1

## Kết quả

| Hạng mục | Kết quả |
|---|---|
| `python -m unittest discover -s tests -v` | **44/44 tests PASS** |
| `python -m scripts.smoke_test` | **PASS**, tất cả CLI con trả exit code 0 |
| Train demo 2 epoch → lưu best/last | PASS |
| Dừng sau epoch 1 → resume epoch 2 | PASS |
| So model sau resume với train liên tục trên CPU, có Dropout/DropPath | Trùng từng tensor, tolerance=0 |
| Evaluate test + predict 1 clip qua checkpoint | PASS |
| Masked pretrain → nạp cả bốn encoder → fine-tune | PASS; demo nạp 52 state tensors/encoder |
| Model template MultiVSL, B=2, T=150, D=128, 199 lớp | Forward + backward PASS, logits/gradient finite |
| Số tham số model MultiVSL template | 3,899,192 |
| Số tham số model demo | 53,264 |
| `27kpt.zip` → trial 10 lớp → check data → train CPU 2 epoch | PASS; 245 mẫu, 166,988 tham số |

## Phạm vi 44 tests

- 17 data/config tests: One Euro alpha/constant/step/timestamps, layout, nội suy, khớp mất, root mất, XYZ vs confidence, valid length, padding, resample endpoints, bone scaling, augmentation và split leakage.
- 11 model tests: bốn stream, gradient các encoder/fusion, mask boundaries, padding không đổi logits, T khác 27, graph support/normalization, load pretrained strict cả bốn stream, reconstruction mask và positional gradient.
- 4 training/metrics tests: accumulation với micro-batch cuối nhỏ, focal gamma=0, Mixup cùng mask, confusion/macro-F1.
- 8 data-utility tests: MultiVSL signer/random split, từ chối signer lạ, VSL conversion trục/length/person, từ chối length mơ hồ, pickle opt-in và chuẩn bị trực tiếp `27kpt.zip`.
- 4 workflow tests: resume chính xác ở epoch boundary, pretrain-fine-tune-save-load, từ chối overwrite, từ chối đổi schedule khi resume.

## Smoke test đã đi qua các CLI

`make_demo_data` → `check_data` (có hash duplicate) → `trace_flow` → `train --stop-after 1` → `train --resume` → `evaluate` → `predict` → `pretrain` → `train --pretrained`.

Dữ liệu smoke gồm 15 train, 6 val, 6 test, 3 lớp giả. Top-1 của run smoke là 1/3; đây là dữ liệu/cấu hình kiểm tra đường chạy, **không chứng minh mô hình học tốt hoặc cải thiện accuracy**. Checkpoint giả không đóng kèm bản ZIP để tránh bị nhầm với model đã train thật; có thể tái tạo bằng script.

## Chưa kiểm thử / chưa kết luận

- Chưa chạy trên toàn bộ MultiVSL200 hoặc VSL400 thật. Đã xác nhận và chạy trial 10 lớp từ gói `27kpt.zip`, nhưng chưa benchmark đủ 199 lớp.
- Chưa có accuracy/F1 nghiên cứu, ablation, so sánh baseline cũ/mới hoặc nhiều seed thật.
- Chưa chạy CUDA/AMP thực, Kaggle, Windows hoặc multi-worker trên các hệ điều hành khác.
- Chưa profiling VRAM/throughput; không bảo đảm batch size template vừa mọi GPU.
- CI workflow đã được thêm nhưng chưa chạy trên GitHub Actions của người dùng.
- Chưa chứng minh tính tương đương với checkpoint/implementation nguồn; đây là clean baseline có thay đổi được ghi rõ.

## Tái chạy

```bash
python -m unittest discover -s tests -v
python -m scripts.smoke_test --output runs/verification_new
```

Không sửa các test để che lỗi khi đổi model. Khi bổ sung nghiên cứu mới, thêm test mask/shape/gradient và benchmark protocol riêng.
