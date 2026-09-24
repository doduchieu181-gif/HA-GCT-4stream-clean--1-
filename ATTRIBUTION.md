# Nguồn tham khảo và phạm vi

- Repository do người dùng cung cấp: [KILIAN0802/HA-GCT](https://github.com/KILIAN0802/HA-GCT).
- Nhánh kiến trúc bốn stream được đối chiếu: [`hagct`](https://github.com/KILIAN0802/HA-GCT/tree/hagct), commit `693a2788492f5a0335a50cf50addadd1a2bf90f4`.
- Topology 27 joint và định nghĩa Joint/Bone/Motion/Bone-Motion theo đường triển khai ở nhánh trên.
- [One Euro Filter — trang tác giả](https://gery.casiez.net/1euro/): tham khảo công thức adaptive low-pass.
- [PyTorch 2.6 AMP examples](https://docs.pytorch.org/docs/2.6/notes/amp_examples.html): tham khảo thứ tự scale/unscale/gradient accumulation.

Code được viết lại để minh bạch dữ liệu, mask, training và tài liệu. Đây không phải tuyên bố phát minh kiến trúc mới, không phải bản sao checkpoint tương đương số học và không phải xác nhận kết quả của bất kỳ paper nào.

Snapshot nguồn được kiểm tra không có file LICENSE ở root; tài liệu này không tự gán MIT/Apache hay tuyên bố quyền tái phân phối phần mã/tài sản của tác giả gốc. Trước khi công bố hoặc phân phối thương mại, xác minh quyền sử dụng và nghĩa vụ trích dẫn của repository, paper và dataset liên quan. Dataset, slide và pretrained weights của bên thứ ba không được đóng gói trong bản bàn giao.
