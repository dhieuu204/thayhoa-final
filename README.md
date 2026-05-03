# Hệ thống Tìm kiếm File Nhạc Theo Nội dung (CBMIR)

## Giới thiệu

Hệ thống tìm kiếm bản nhạc dựa trên nội dung âm thanh (Content-Based Music Information Retrieval). Đầu vào là một file âm thanh, đầu ra là 5 bản nhạc giống nhất trong cơ sở dữ liệu, xếp theo thứ tự giảm dần về độ tương đồng.

---

## 1. Bộ dữ liệu

### Thu thập
- **Số lượng**: ≥ 500 files âm thanh nhạc không lời (beat karaoke)
- **Nguồn**: YouTube (tìm kiếm tự động theo tên bài)
- **Công cụ**: `yt-dlp` + `ffmpeg`

### Chuẩn hóa
Tất cả file sau khi tải về đều được chuẩn hóa về cùng một định dạng:

| Thuộc tính | Giá trị |
|---|---|
| Định dạng | WAV |
| Tần số lấy mẫu | 44100 Hz |
| Mức lượng tử hóa | 16-bit PCM |
| Kênh âm thanh | Mono (1 kênh) |
| Độ dài | 60 giây |

> **Lý do chuẩn hóa**: Các đặc trưng như ZCR, Energy phụ thuộc trực tiếp vào số mẫu trong 1 giây. Nếu sample rate khác nhau, cùng 1 bài nhạc sẽ cho ra vector đặc trưng khác nhau → so sánh sai.

### Cách tải dữ liệu
```bash
# Tạo file danh sách bài hát (mỗi dòng 1 tên)
# Nhac_Tre.txt, Nhac_Vang.txt, ...

python download_beat.py Nhac_Tre.txt
# → Lưu vào dataset_beat/nhac_tre_TenBaiHat.wav
```

---

## 2. Bộ Đặc trưng (Feature Vector)

Mỗi file nhạc được biểu diễn bằng **vector 18 chiều**:

```
V = [pitch, zcr, energy, centroid, bandwidth, mfcc_1, mfcc_2, ..., mfcc_13]
```

### Chi tiết từng đặc trưng

| # | Đặc trưng | Ý nghĩa | Lý do chọn |
|---|---|---|---|
| 1 | **Pitch (F0)** | Tần số cơ bản trung bình (Hz) | Xác định tông/giai điệu của bài nhạc |
| 2 | **ZCR** | Zero-Crossing Rate — tốc độ đổi dấu tín hiệu | Phân biệt nhạc có nhịp nhanh/chậm, beat nhiều/ít |
| 3 | **Energy** | Năng lượng trung bình `mean(x²)` | Phân biệt nhạc mạnh/nhẹ, to/nhỏ |
| 4 | **Spectral Centroid** | Trọng tâm phổ (Hz) | Âm thanh sáng/chói hay ấm/trầm |
| 5 | **Spectral Bandwidth** | Băng thông phổ | Dải âm rộng (orchestra) hay hẹp (solo) |
| 6-18 | **MFCC 1-13** | Mel-Frequency Cepstral Coefficients | Biểu diễn toàn bộ âm sắc của nhạc cụ |

### Tại sao dùng kết hợp nhiều đặc trưng?

Mỗi đặc trưng đơn lẻ chỉ mô tả được một khía cạnh của âm nhạc. Kết hợp thành vector giúp mô tả toàn diện hơn:

```
Pitch    → "bài này ở tông cao hay thấp?"
ZCR      → "nhịp nhanh hay chậm?"
Energy   → "nhạc to hay nhỏ?"
Centroid → "âm thanh sáng hay tối?"
MFCC     → "nhạc cụ này nghe như thế nào?"
──────────────────────────────────────────
Kết hợp → nhận diện chính xác từng bản nhạc
```

### Trích xuất đặc trưng
```python
import librosa
import numpy as np

audio, sr = librosa.load("bainhat.wav", sr=44100, mono=True)

# 1. Pitch
f0, voiced_flag, _ = librosa.pyin(audio,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"))
pitch = np.nanmean(f0[voiced_flag])

# 2. ZCR
zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

# 3. Energy
energy = np.mean(audio ** 2)

# 4. Spectral Centroid
centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

# 5. Bandwidth
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

# 6. MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfcc_mean = mfcc.mean(axis=1)

vector = [pitch, zcr, energy, centroid, bandwidth] + mfcc_mean.tolist()
# → vector 18 chiều
```

---

## 3. Cơ sở dữ liệu và Cơ chế Tìm kiếm

### Cấu trúc CSDL (SQLite)

```sql
CREATE TABLE songs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT,        -- tên bài hát
    file_path TEXT,        -- đường dẫn file WAV
    pitch     REAL,        -- giá trị pitch (để thống kê)
    zcr       REAL,
    energy    REAL,
    centroid  REAL,
    bandwidth REAL,
    features  BLOB         -- toàn bộ vector 18 chiều (pickle)
)
```

### Tổ chức chỉ số — KD-Tree

500 vector 18 chiều được tổ chức thành cây KD-Tree:

```
                 [Chia theo chiều pitch]
                    pitch < 130Hz?
                   /              \
         [pitch < 100Hz]      [pitch ≥ 130Hz]
          /        \            /           \
    [energy<0.01] ...    [zcr < 0.03]   [zcr ≥ 0.03]
        ...                 ...              ...
```

- **Brute-force**: so sánh với 500 bài → O(n)
- **KD-Tree**: chỉ đi theo ~log₂(500) ≈ 9 nhánh → O(log n)

### Đo khoảng cách

**L2 (Euclidean)**:
```
dist(Q, D) = sqrt((q1-d1)² + (q2-d2)² + ... + (q18-d18)²)
```

**L1 (Manhattan)**:
```
dist(Q, D) = |q1-d1| + |q2-d2| + ... + |q18-d18|
```

Khoảng cách càng nhỏ → hai bản nhạc càng giống nhau.

---

## 4. Hệ thống Tìm kiếm

### Sơ đồ khối

```
┌─────────────────────────────────────────────────────────┐
│                   LUỒNG TẠO CSDL                        │
│                                                         │
│  500 file WAV → Chuẩn hóa → Trích xuất vector          │
│              → Lưu SQLite + Build KD-Tree               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  LUỒNG TRUY VẤN                         │
│                                                         │
│  File query.wav                                         │
│       ↓ Chuẩn hóa (44100Hz, mono, 16-bit)              │
│       ↓ Trích xuất vector Q (18 chiều)                  │
│       ↓ KD-Tree.query(Q, k=5, p=2)  [L2]               │
│       ↓ Tính khoảng cách + sắp xếp                     │
│       ↓ Chuyển sang % tương đồng                        │
│  Top 5 bài giống nhất (giảm dần)                        │
└─────────────────────────────────────────────────────────┘
```

### Kết quả trung gian

```
[1] Vector query Q = [130.9, 0.015, 0.013, 2150.3, 1820.5, -210.3, ...]

[2] Bảng khoảng cách:
    #1  dist =  0.000  → Hoa Hải Đường       ← chính nó
    #2  dist = 12.400  → Chỉ Là Không Cùng Nhau
    #3  dist = 45.800  → Chạy Ngay Đi
    #4  dist = 67.200  → Nơi Này Có Anh
    #5  dist = 89.200  → Có Chắc Yêu Là Đây

[3] Kết quả cuối (% tương đồng):
    #1  100.00%  Hoa Hải Đường
    #2   86.10%  Chỉ Là Không Cùng Nhau
    #3   48.70%  Chạy Ngay Đi
    #4   24.50%  Nơi Này Có Anh
    #5    0.00%  Có Chắc Yêu Là Đây
```

### Cách chạy

```bash
# Build database từ dataset
python main.py build

# Tìm kiếm (L2 mặc định)
python main.py search dataset_beat/Hoa_Hải_Đường.wav

# Tìm kiếm với L1
python main.py search dataset_beat/Hoa_Hải_Đường.wav l1
```

---

## 5. Demo và Đánh giá

### Demo Case 1 — File có trong DB

```bash
python main.py demo_in dataset_beat/Hoa_Hải_Đường.wav
```

Kết quả mong đợi: Top 1 là chính file đó với khoảng cách = 0.

### Demo Case 2 — File ngoài DB

```bash
python main.py demo_out query.wav "Hoa Hải Đường,Nơi Này Có Anh"
```

Kết quả: 5 bài gần nhất trong DB với file chưa từng được index.

### Đánh giá Precision & Recall

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| **Precision** | ca / (ca + fa) | Trong 5 kết quả, bao nhiêu % đúng? |
| **Recall** | ca / (ca + fd) | Bao nhiêu % bài đúng được tìm thấy? |

```
ca = Correct Accepted  (lấy đúng)
fa = False Accepted    (lấy sai)
fd = False Dismissed   (đúng nhưng bỏ sót)
```

### Benchmark tốc độ

```bash
python main.py bench dataset_beat/Hoa_Hải_Đường.wav
```

```
Phương pháp          Thời gian TB
------------------------------------
Brute-force              45.000 ms
KD-Tree                   2.100 ms
------------------------------------
KD-Tree nhanh hơn : 21.4x
```

---

## Cấu trúc dự án

```
thayhoa/
├── download_beat.py       # Tải beat karaoke từ YouTube
├── feature_extraction.py  # Trích xuất vector + build DB + KD-Tree
├── search.py              # Engine tìm kiếm Top 5
├── evaluate.py            # Precision, Recall, Benchmark
├── main.py                # Giao diện chạy tất cả
├── test.txt               # Danh sách bài hát mẫu
├── dataset_beat/          # Thư mục chứa file WAV (không commit)
├── music_database.db      # SQLite database (không commit)
└── kdtree.pkl             # KD-Tree đã build (không commit)
```

## Yêu cầu

```bash
pip install yt-dlp librosa numpy scipy scikit-learn
sudo apt install ffmpeg
```
