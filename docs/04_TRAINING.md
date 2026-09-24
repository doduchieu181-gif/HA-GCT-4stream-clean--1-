# 04 — Huấn luyện, checkpoint và đánh giá

## 1. Cấu hình mặc định quan trọng

| Tham số | Mặc định | Vai trò |
|---|---|---|
| `train.batch_size` | 8 | Micro-batch đi qua model một lần |
| `train.accum_steps` | 4 | Số micro-batch trước optimizer update |
| `train.lr` | 0.0003 | AdamW base learning rate |
| `train.classifier_lr_multiplier` | 1.0 | Hệ số LR của bốn classifier; fusion weights dùng base LR |
| `train.weight_decay` | 0.05 | Áp dụng cho mọi tham số trong baseline này, kể cả bias/norm |
| `train.warmup_epochs` | 5 | Warmup theo số optimizer update |
| `train.min_lr_ratio` | 0.01 | Tỉ lệ LR cuối so với base LR |
| `train.loss` | ce | `ce` hoặc `focal` |
| `train.label_smoothing` | 0.1 | Label smoothing trong CE |
| `train.focal_gamma` | 2.0 | Exponent của `(1-p_true)^gamma` khi dùng focal |
| `train.amp` | true | Mixed precision chỉ trên CUDA |
| `train.clip_grad` | 1.0 | Max norm sau unscale và normalize gradient |
| `train.patience` | 20 | Dừng sau từng này epoch val không cải thiện |
| `train.balanced_sampling` | false | WeightedRandomSampler, chỉ supervised train |
| `train.mixup_alpha` | 0.0 | Mặc định tắt để có baseline dễ kiểm soát |

Effective batch thường gần `batch_size × accum_steps`, nhưng nhóm cuối có thể nhỏ hơn. Gradient được chuẩn hóa theo **số mẫu thật của nhóm**, không luôn chia cố định cho `accum_steps`.

Scheduler có số bước tính từ dataset và epochs, warmup được giới hạn không vượt tổng số bước trừ 1. `scheduler.step()` chỉ chạy sau optimizer update thành công; khi AMP bỏ qua bước do overflow, không tiến lịch LR. Log `lr_next` là LR cho update tiếp theo, không phải LR trung bình epoch vừa chạy.

## 2. Loss, sampling và Mixup

CE nhận logits thô và label index. Focal dùng `p_true` từ log-softmax với nhãn thật, nhân với CE có thể đã label-smoothing. Đặt `label_smoothing=0` nếu muốn focal dạng không smoothing; gamma=0 bằng CE.

Balanced sampler lấy xác suất mỗi mẫu tỉ lệ nghịch số mẫu của lớp, lấy có hoàn lại `len(train_set)` mẫu/epoch. Không đồng thời tự thêm class weights vào loss. Cùng sample được lấy lặp trong cùng epoch có cùng augmentation vì seed dựa trên `(seed, epoch, index)`.

Mixup chỉ ghép các sample có **mask giống hoàn toàn**, giữ nguyên ý nghĩa frame hợp lệ. Với clip dài/ngắn và random placement đa dạng, ít cặp có mask trùng nên hiệu quả mixup có thể thấp. Đây là baseline an toàn về mask, không phải temporal alignment Mixup. Loss train khi bật Mixup là loss nhãn trộn; không log train accuracy có thể gây hiểu nhầm.

## 3. Pretrain và fine-tune

```bash
python pretrain.py --config configs/multivsl200.json --output runs/pretrain_01 --epochs 50
python train.py --config configs/multivsl200.json --output runs/finetune_01 --pretrained runs/pretrain_01/best.pt
```

Pretrain bỏ qua `train.loss`, `label_smoothing`, `focal_gamma`, `mixup_alpha`, `balanced_sampling`, `classifier_lr_multiplier`; dùng masked reconstruction MSE. Split labels vẫn phải đúng schema manifest nhưng không được đưa vào reconstruction loss. Nhãn val cũng không được dùng trong loss pretrain.

Kiến trúc encoder (D, layers, heads) và `max_frames` phải tương thích. Load strict theo key/shape; in số tensor nạp cho **từng stream**. Không bỏ qua im lặng khi tên key khác. `--pretrained` chỉ nhận checkpoint pretrain do project này tạo, **không nhận checkpoint repo cũ** hay supervised checkpoint.

Pretrain/fine-tune có thể dùng số lớp khác vì encoder không chứa classifier, nhưng phải kiểm tra semantics joint/channel/preprocessing của hai dataset. Code kiểm tra kiến trúc, **không chứng minh hai dữ liệu có cùng ý nghĩa**.

## 4. File đầu ra

| File | Nội dung |
|---|---|
| `config.json` | Toàn bộ cấu hình thực dùng, kể cả default và CLI override |
| `history.jsonl` | Một JSON/epoch: train loss, val metrics, LR tiếp theo, thời gian, fusion weights |
| `best.pt` | Val top-1 cao nhất; pretrain là val MSE thấp nhất |
| `last.pt` | Epoch hoàn thành gần nhất để resume |

Checkpoint có version schema, kind, config, SHA256 **nội dung manifest CSV**, model, optimizer, scheduler, AMP scaler, epoch/best/bad_epochs, CPU/CUDA/Python RNG. Hash CSV **không hash toàn bộ file tensor**; sửa `.npy` giữ nguyên manifest không được tự phát hiện.

`weights_only=True` khi load giúp tránh tùy ý giải tuần tự object Python; chỉ dùng checkpoint từ nguồn tin cậy. Không tự vô hiệu hóa cơ chế này để nạp file lạ.

Run mới không ghi đè thư mục có dữ liệu. Best/last cập nhật trong **chính run đã được tạo** bằng ghi tạm + replace. Không có thao tác sửa repository nguồn hoặc push GitHub.

## 5. Resume đúng cách

```bash
# Tổng lịch 100 epoch giữ nguyên; dừng chủ động sau 5 epoch:
python train.py --config configs/multivsl200.json --stop-after 5
python train.py --config configs/multivsl200.json --resume runs/multivsl_baseline/last.pt
```

`--stop-after` tính theo số epoch tổng đã hoàn thành, không phải số epoch thêm. `history.jsonl` ghi epoch zero-based.

Resume cho phép đổi device, num_workers và biểu diễn output_dir nhưng checkpoint vẫn phải nằm trong output dir đang chạy; không cho đổi hyperparameter, manifest bytes hoặc tổng epochs. Nếu dùng CLI override `--epochs`/`--output` lúc đầu, cần cung cấp cùng giá trị khi resume. Dùng `last.pt` mới nhất; không quay lại `best.pt` cũ trong log có epoch mới hơn.

Khôi phục ở **ranh giới epoch**: ngắt giữa epoch sẽ chạy lại epoch đó từ last.pt. Test CPU xác minh trùng từng tensor khi resume; **không hứa bitwise determinism CUDA**, khác phiên bản thư viện/thiết bị hoặc thay nguồn dữ liệu. Nếu chưa hoàn thành epoch đầu, chưa có checkpoint để resume. Run đã early-stop sẽ không tiếp tục optimizer khi resume.

## 6. Đánh giá và suy luận

```bash
python evaluate.py --checkpoint runs/finetune_01/best.pt --split test --output runs/finetune_01_test
python predict.py --checkpoint runs/finetune_01/best.pt --input /path/to/clip.npy --classes data/multivsl/classes.json
```

Evaluate dùng preprocessing trong checkpoint, tắt augmentation, không balanced sampler. Nếu chuyển máy, dùng `--manifest /new/path/manifest.csv`. Không cần train config riêng cho evaluate. Output mới gồm:

- `metrics.json`: top-1, top-min(5,K), macro-F1, số mẫu, fusion weights.
- `per_class.csv`: support, precision, recall, F1 theo ID.
- `confusion.npy`: hàng = nhãn thật, cột = dự đoán.
- `predictions.csv`, `logits.npy`: cùng thứ tự manifest của split được chọn.

Macro-F1 tính trên **tất cả K lớp**, lớp không support/prediction có F1=0. Có thể khác convention một thư viện khác; luôn ghi rõ khi báo cáo. Demo K=3 nên metric tên `top3`, không giả gọi là top5.

Predict trả softmax score, top-k và top-1 riêng từng stream. Score **chưa được calibration** và không phải xác suất đáng tin cậy ngoài phân phối train. Dữ liệu mới phải có cùng joint order, channel semantics và đơn vị/preprocessing phù hợp.

## 7. Ablation tối thiểu trước khi kết luận cải tiến

Giữ nguyên split, seed set, số epoch/budget; chạy baseline không pretrain, sau đó chỉ đổi một yếu tố: pretrain, filtering, scale normalization, augmentation, focal/balanced sampler. Báo cáo nhiều seed và thời gian/VRAM, không chỉ điểm cao nhất. Single-stream/three-stream ablation chưa có CLI riêng; hiện project chủ đích giữ một luồng bốn stream dễ hiểu.
