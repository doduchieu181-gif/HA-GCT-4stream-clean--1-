# 02 — Dữ liệu, split và topology

## 1. Hợp đồng đầu vào

Mỗi clip là một file `.npy` số thực, không chứa Python object. Chọn rõ:

| Cấu hình | Ý nghĩa |
|---|---|
| `layout=TVC` | `(frame, joint, channel)` |
| `layout=CTV` | `(channel, frame, joint)` |
| `channels=xy` | 2 channel tọa độ |
| `channels=xy_conf` | 3 channel: X, Y, confidence trong [0,1] |
| `channels=xyz` | 3 channel tọa độ; bản 2D này **bỏ Z**, không dùng Z làm confidence |

Số joint bắt buộc là **27**, theo thứ tự bên dưới. Không tự đoán layout/channels, không tự cắt 27 khớp đầu từ skeleton khác. Không có tọa độ 3D trong model này.

`fps` cần tương ứng tốc độ lấy mẫu đầu vào. Pipeline hiện giả sử frame cách đều và dùng `fps` cố định; hàm One Euro riêng có hỗ trợ timestamps nhưng Dataset chưa nhận timestamps biến thiên.

## 2. Manifest CSV

```csv
path,label,split,signer,length
samples/word001_S01_01.npy,0,train,S01,47
samples/word001_S02_01.npy,0,val,S02,39
samples/word001_S03_01.npy,0,test,S03,42
```

- `path`: tương đối với **thư mục chứa manifest**, hoặc absolute path.
- `label`: số nguyên **0..K-1**. Không tự remap trong lúc train. `num_classes=K`.
- `split`: `train`, `val`, `test`. Train cần train + val; test dùng khi evaluate.
- `signer`: bắt buộc nếu `signer_disjoint=true`. Khi bật, một signer chỉ thuộc một split.
- `length`: tùy chọn. Nếu raw đã right-pad, khai báo số frame thật **ở đầu clip**. File chưa pad thì để trống/bỏ cột. Nếu dữ liệu pad ở đầu/giữa, cần cắt đúng vùng hợp lệ trước khi đưa vào pipeline này.

`classes.json` là list tên theo thứ tự label, ví dụ `["xin_chao", "cam_on"]`. Train chỉ dùng ID; predict dùng file này nếu cung cấp. File phải đi cùng manifest và đúng thứ tự lớp; **checkpoint không tự xác minh tên lớp**.

Đường dẫn `data.manifest` trong config được tính từ working directory, không phải thư mục `configs/`. Luôn chạy CLI tại project root hoặc dùng absolute path.

## 3. Thứ tự 27 joint

| ID | Khớp | Parent |
|---|---|---|
| 0 | Nose/root | 0 |
| 1, 2 | Vai trái, phải | 0, 0 |
| 3, 4 | Khuỷu trái, phải | 1, 2 |
| 5, 6 | Cổ tay trái, phải | 3, 4 |
| 7 | Lòng bàn tay trái | 5 |
| 8 | Ngón cái trái | 7 |
| 9, 10 | Gốc/đầu ngón trỏ trái | 7, 9 |
| 11, 12 | Gốc/đầu ngón giữa trái | 7, 11 |
| 13, 14 | Gốc/đầu ngón áp út trái | 7, 13 |
| 15, 16 | Gốc/đầu ngón út trái | 7, 15 |
| 17 | Lòng bàn tay phải | 6 |
| 18 | Ngón cái phải | 17 |
| 19, 20 | Gốc/đầu ngón trỏ phải | 17, 19 |
| 21, 22 | Gốc/đầu ngón giữa phải | 17, 21 |
| 23, 24 | Gốc/đầu ngón áp út phải | 17, 23 |
| 25, 26 | Gốc/đầu ngón út phải | 17, 25 |

Định nghĩa máy đọc được: `hagct/topology.py`. Đây là thứ tự `sign_27` trong nhánh tham khảo, **không phải** thứ tự MediaPipe gốc. Cần giữ quy ước trái/phải của extractor; không có augmentation lật trái/phải trong bản này vì có thể đổi nghĩa ký hiệu.

## 4. Thiếu khớp, chuẩn hóa, padding

Với XY-confidence, điểm confidence thấp hoặc XY không finite được nội suy tuyến tính theo thời gian; ở hai đầu lấy điểm gần nhất. Joint mất toàn clip được đặt về 0 sau normalization. Nếu root mất toàn clip hoặc hai vai không đủ để tính scale, báo lỗi. Muốn ablation bỏ scale, đặt `scale_normalization=false`.

Với XY/XYZ, điểm mất chỉ được xác định qua NaN/Inf, hoặc thêm `(x,y)=(0,0)` nếu bật `zero_is_missing`. Không bật cờ này cho dữ liệu mà `(0,0)` là tọa độ thật hợp lệ, đặc biệt dữ liệu root-centered sẵn.

Lọc One Euro trên XY đã nội suy, reset state mỗi clip. Trừ tọa độ root mỗi frame; chia một scale/clip bằng median khoảng cách hai vai. Clip dài hơn `max_frames` được resample **toàn clip**, không lấy riêng phần đầu. Clip ngắn giữ độ dài; pad sau augmentation.

Joint mất toàn clip giữ 0 là baseline đơn giản, **không phải thuật toán tái tạo giải phẫu**. Bone của khớp mất vẫn có thể không bằng 0 nếu parent còn tọa độ; chưa có per-joint validity gating. Hãy kiểm tra chất lượng extractor và tỷ lệ mất khớp trước benchmark.

## 5. Chuẩn bị MultiVSL

Utility giả định cây `source/<word_id>/<signer>_<clip>.npy` đã chứa đúng 27 joint. Nó lấy signer là phần trước dấu `_` đầu tiên; nếu tên của bạn khác quy ước, chỉnh parser hoặc tạo manifest thủ công.

```bash
python -m scripts.prepare_multivsl --source /path/to/skeletons --output data/multivsl --protocol signer --train-signers S01 S02 S03 --val-signers S04 --test-signers S05
```

Các mã signer trên chỉ là **ví dụ**, không phải protocol chính thức. Utility từ chối signer chưa khai báo hoặc danh sách signer trùng. `classes.json` được sắp theo tên thư mục dạng chuỗi; dùng cùng mapping cho mọi split. Dữ liệu gốc không bị di chuyển.

Nếu nghiên cứu dùng random stratified split, có `--protocol random --seed 42 --val-ratio 0.15 --test-ratio 0.15`. Khi đó đặt `signer_disjoint=false` và ghi rõ không phải cross-signer. Ưu tiên split được benchmark/đề tài quy định hơn utility tự chia.

Template `multivsl200.json` để 199 lớp theo thiết lập nguồn tham khảo; hãy dùng **số lớp thực tế mà utility báo**, không suy ra 200 chỉ từ tên dataset. Template cũng không bảo đảm dữ liệu của bạn là XY-confidence.

## 6. Chuyển VSL400 dạng tensor nhiều mẫu

Utility nhận `.npy` `NCTV`, `NTVC`, hoặc `NCTVM`; labels `.json`/`.npy` số nguyên, hoặc legacy `.pkl` tin cậy. Nó ghi từng mẫu dạng **TVC**, giữ toàn bộ channel khai báo.

```bash
python -m scripts.convert_vsl400 --data /data/train_data.npy --labels /data/train_labels.json --split train --layout NCTV --channels xy --lengths /data/train_lengths.npy --num-classes 400 --output data/vsl400
```

Chạy tương tự với val/test vào cùng output; mỗi split chỉ ghi một lần. Nếu chắc chắn **mọi frame là thật**, thay `--lengths ...` bằng `--all-frames-valid`. Không đoán padding từ các dòng 0. Với `NCTVM`, thêm `--person-index 0` và xác minh người được chọn.

Pickle có thể thực thi code: chỉ thêm `--trust-pickle` với file bạn tin cậy. Dạng hỗ trợ: tuple `(sample_names, labels)` hoặc list nhãn số; không hỗ trợ mọi cấu trúc pickle tùy ý.

Nếu dữ liệu raw không phải 27 joint, phải chuyển topology trước. Không có cam kết mọi phiên bản MultiVSL/VSL400 đều dùng cùng định dạng. Config template là điểm bắt đầu, không phải tự phát hiện dữ liệu.

## 7. Kiểm tra trước khi tốn GPU

### Gói `27kpt.zip` đã xử lý sẵn

Utility `scripts.prepare_27kpt_zip` đọc trực tiếp `raw_npy/labels.csv`, chỉ giải nén sample có nhãn, lấy signer từ tên file và tạo split signer-disjoint. Gói được kiểm tra có 5.223 `.npy` nhưng CSV chỉ ánh xạ 4.791 mẫu; 432 file không có annotation bị bỏ thay vì đoán nhãn. Nhãn 0..198 được giữ nguyên khi dùng đủ 199 lớp; trial N lớp lấy N nhãn nhỏ nhất và remap về 0..N-1, mapping được ghi trong `split_info.json`.

```bash
python -m scripts.check_data --config configs/multivsl200.json --check-duplicates
```

Kiểm tra tất cả mẫu, khoảng valid frames, lớp bị thiếu và tùy chọn bản sao giống từng byte. Manifest kiểm tra trùng đường dẫn và signer; hash byte không phát hiện clip gần giống hoặc một người mang hai ID. Chống rò rỉ vẫn cần kiểm soát nguồn dữ liệu và protocol.
