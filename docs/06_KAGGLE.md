# 06 — Chạy trên Kaggle/notebook và xử lý lỗi

## 1. Bắt đầu trên notebook

Giải nén project vào thư mục **ghi được** (ví dụ `/kaggle/working/ha-gct-4stream-clean`). Dữ liệu có thể nằm trong `/kaggle/input/...` chỉ đọc. Không ghi run/checkpoint vào thư mục input.

Trong notebook:

```python
%cd /kaggle/working/ha-gct-4stream-clean
import torch
print(torch.__version__, torch.cuda.is_available())
```

Nếu PyTorch đã có và >=2.6, không cài bản CPU đè lên CUDA. Chỉ cài dependencies thiếu. Nếu cần thay bản PyTorch, dùng hướng dẫn chính thức tương thích CUDA và có thể phải restart kernel.

```python
!pip install -r requirements.txt
!python -m scripts.smoke_test --output runs/smoke_notebook
```

## 2. Chuẩn bị config thật

Các config là JSON thông thường. Có thể tạo bản riêng bằng cell:

```python
import json
from pathlib import Path

cfg = json.loads(Path("configs/multivsl200.json").read_text())
cfg["data"]["manifest"] = "/kaggle/working/my_manifest/manifest.csv"
# Sửa theo dữ liệu THỰC, không sao chép mù các giá trị sau:
cfg["data"]["layout"] = "TVC"
cfg["data"]["channels"] = "xy_conf"
cfg["model"]["num_classes"] = 199
cfg["train"]["device"] = "cuda"
cfg["train"]["batch_size"] = 4
cfg["train"]["accum_steps"] = 8
cfg["train"]["num_workers"] = 0
cfg["train"]["output_dir"] = "runs/my_baseline"
Path("configs/my_run.json").write_text(json.dumps(cfg, indent=2))
```

`num_workers=0` trước để lỗi sample hiện rõ. Sau khi pipeline đúng, có thể tăng 2/4 và đo RAM/tốc độ. `data.cache=true` giữ dữ liệu đã preprocess trong RAM, không bật nếu dataset vượt bộ nhớ.

```python
!python -m scripts.check_data --config configs/my_run.json
!python train.py --config configs/my_run.json
!python evaluate.py --checkpoint runs/my_baseline/best.pt --split test --output runs/my_baseline_test
```

Để tải kết quả về máy: lưu `configs/my_run.json`, manifest/classes mapping và thư mục `runs/my_baseline`. Không cần chia sẻ dataset nếu không có quyền. Checkpoint có đường dẫn manifest cũ; chuyển máy dùng `evaluate.py --manifest ...`.

## 3. Lỗi thường gặp

| Lỗi/hiện tượng | Kiểm tra |
|---|---|
| `ModuleNotFoundError: hagct` | Chạy ở project root; tiện ích dùng `python -m scripts.<tên>`, không gọi `python scripts/file.py` |
| `Cần TVC=(T,27,...)` | Đúng layout? Đã chọn đúng 27 joint? Có thừa chiều person? |
| Confidence ngoài [0,1] | Channel thứ ba là Z hay confidence? Confidence có đang ở thang 0..100 không? |
| Root mất toàn clip | Kiểm tra extractor/threshold/root index; không tự bỏ lỗi bằng zero-fill |
| Khoảng cách hai vai bằng 0 | Kiểm tra topology/đơn vị; dữ liệu không hỗ trợ scale vai thì tắt scale có chủ đích |
| Train thiếu lớp | `num_classes`/mapping/split có sai? Không tự bỏ lớp để qua kiểm tra |
| Signer trùng split | Sửa protocol/manifest; chỉ tắt signer_disjoint nếu thực sự dùng protocol không cross-signer |
| File đã tồn tại | Chọn output mới hoặc resume đúng last.pt; không cần xóa run cũ |
| CUDA OOM | Giảm micro-batch; giảm T/D/layers là thay đổi mô hình phải ghi trong báo cáo |
| CPU chạy chậm | Demo thêm `--threads 1`; trên máy thật đo rồi chọn threads phù hợp |
| Pretrained mismatch | Cùng D, layers, heads, Tmax và checkpoint do bản clean tạo? |
| Resume đổi epochs bị từ chối | Không kéo dài lịch cosine cũ âm thầm; giữ epochs cũ hoặc thiết kế run mới |
| Accuracy quanh ngẫu nhiên | Xem confusion/nhãn/split/preprocessing trước; demo chỉ là dữ liệu giả |

Môi trường GPU/Kaggle thật chưa được kiểm thử ở bản bàn giao. Đường dẫn và accelerator tùy môi trường của bạn; không có cam kết một batch size cố định luôn vừa VRAM.
