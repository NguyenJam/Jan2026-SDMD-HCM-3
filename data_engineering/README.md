# Phần Data Engineer của Vũ Lê Hoàng

Bộ tài liệu và mã nguồn cho Group Task 1, nhóm HCM-3, MSSV **104774231**. Phần này hoàn thành việc kiểm tra dữ liệu, phân tích CMUdict và chuẩn bị parser để bàn giao cho nhóm. Mã nguồn chạy bằng **Python 3.10 trở lên**, chỉ dùng thư viện chuẩn, không cần cài package.

## Mở các file này trước

- [DATA_ENGINEERING_REPORT.md](DATA_ENGINEERING_REPORT.md): báo cáo tiếng Anh về nguồn, số liệu, parsing, giới hạn và bàn giao.
- [GROUP_REPORT_UPDATES.md](GROUP_REPORT_UPDATES.md): các đoạn tiếng Anh để đưa vào báo cáo nhóm, gồm bảng số liệu, trạng thái thực hiện, contribution và AI declaration.
- [outputs/verification_summary.md](outputs/verification_summary.md): kết quả phân tích và kiểm chứng.
- [outputs/test_results.txt](outputs/test_results.txt): log kiểm thử thực tế.

## Kết quả chính

| Chỉ số | Bản đã kiểm chứng | Báo cáo nhóm đang ghi |
|---|---:|---:|
| Mục phát âm | 133.854 | 134.373 |
| Từ gốc | 125.074 | 125.770 |
| Từ có nhiều mục phát âm | 8.154 | 8.016 |
| Âm vị cơ sở | 39 | 39 |

Bản đã kiểm chứng là **CMUdict 0.7b, SVN revision 12857**, có checksum trong [source_lock.json](source_lock.json). Chưa có file mà nhóm đã dùng để tạo các con số cũ, nên chưa thể kết luận vì sao lệch. Nếu nhóm chọn bản đi kèm này thì dùng số liệu đã kiểm chứng và nguồn tương ứng; không trộn số liệu giữa các bản.

File từ điển cần đọc bằng **Latin-1** để giữ mục `DÉJÀ`. Có **69 ký hiệu xuất hiện thực tế** khi tính cả chữ số trọng âm; bảng đi kèm cho phép **84 ký hiệu**, và có **39 âm vị cơ sở**. Đây là ba cách đếm khác nhau.

## Chạy lại toàn bộ

Từ thư mục gốc của repository nhóm, mở PowerShell:

```powershell
Set-Location -LiteralPath '.\data_engineering'
python verify_work.py
```

Lệnh trên không dùng mạng. Nó kiểm tra checksum, phân tích toàn bộ dữ liệu, chạy 14 ca tra cứu và 20 test, rồi lưu kết quả vào `outputs/`. Lệnh trả exit code khác 0 nếu có lỗi. Có thể chạy từ thư mục khác bằng cách truyền đường dẫn đầy đủ đến `verify_work.py`.

Các lệnh riêng:

```powershell
python fetch_data.py --verify-only
python run_analysis.py
python -m unittest discover -s tests -v
python run_analysis.py --lookup read
python run_analysis.py --lookup read --variant 1
```

Ba file nguồn đã được tải kèm. Nếu chuyển mã sang máy khác mà thiếu nguồn, chạy `python fetch_data.py` khi có mạng. Script chỉ chấp nhận byte khớp checksum, và sẽ không ghi đè file hiện có bị thay đổi.

## Cách bàn giao cho Khanh và Duy

```python
from cmudict_data import Lexicon
from fetch_data import ROOT, verify_sources

verify_sources()
lexicon = Lexicon.from_file(ROOT / 'source' / 'cmudict-0.7b')
result = lexicon.lookup('read')               # Trả hai biến thể, chưa chọn thay người dùng.
result = lexicon.lookup('read', variant_id=1) # R IY1 D

if result['status'] == 'known' and result['selected'] is not None:
    phones = result['selected']['phones']    # Bàn giao chuỗi này cho adapter của TTS.
```

- **Khanh:** dùng [outputs/phone_inventory.csv](outputs/phone_inventory.csv) để xây ánh xạ sang bộ âm vị của mô hình. Bảng có cả ký hiệu hợp lệ nhưng chưa xuất hiện, với số lần xuất hiện bằng 0. Parser chưa kiểm tra mô hình TTS nào.
- **Duy:** dùng [regression_cases.json](regression_cases.json) và [outputs/lookup_evidence.json](outputs/lookup_evidence.json) làm ca kiểm thử. `oov`, `invalid_input`, `invalid_variant` hoặc chưa chọn biến thể đều không có phát âm được chọn. Duy quyết định chính sách tách câu/cụm từ ở bước tích hợp; hàm này chỉ nhận một token từ điển mỗi lần.
- **Đăng:** nội dung Jira/Confluence có thể lấy từ `GROUP_REPORT_UPDATES.md`. Trạng thái tích hợp toàn hệ thống vẫn là đang chờ.

Các ID biến thể giữ số trong nguồn: mục không có hậu tố dùng ID `0`, mục `(1)` dùng ID `1`. Không được suy ra ID này là thì, nghĩa hay độ phổ biến của từ.

## Phạm vi đã hoàn thành

- Lấy và cố định ba file từ nguồn CMUSphinx chính thức; giữ nguyên nguồn và bản quyền.
- Kiểm chứng số liệu, encoding, biến thể, âm vị và cấu trúc toàn bộ từ điển.
- Chuẩn bị parser, API tra cứu một token và minh chứng kiểm thử.
- Soạn báo cáo, nội dung cập nhật báo cáo nhóm và phần đóng góp.

Phần tạo WAV, ánh xạ sang mô hình cụ thể, UI, tokenization cho cụm từ và CI của nhóm vẫn cần các thành viên tích hợp. Đây là mã hỗ trợ phần dữ liệu, chưa phải sản phẩm TTS hoàn chỉnh.

Báo cáo `Group-Task-1.docx` do nhóm quản lý riêng và không được chỉnh sửa trong thay đổi này. Các câu nói dự án hoàn toàn chưa có code cần đổi theo đoạn trạng thái trong `GROUP_REPORT_UPDATES.md` nếu nhóm đưa mã này vào bài. AI declaration cũng đã được soạn lại để phản ánh việc có AI hỗ trợ mã nguồn và kiểm thử.
