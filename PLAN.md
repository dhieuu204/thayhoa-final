# Kế hoạch sửa đổi: Chuyển sang Z-score + L2

## Bối cảnh

Hệ thống hiện đang dùng **Unit norm + Cosine Similarity** để đo độ tương đồng.
Vấn đề: tất cả kết quả similarity đều nằm trong khoảng `0.999x`, khó phân biệt trực quan.

**Nguyên nhân:** 18 đặc trưng có đơn vị rất khác nhau:
- Pitch: 0 – 800 Hz
- ZCR: 0.00001 – 0.0002
- MFCC: -500 đến +200

Unit norm được dùng để triệt tiêu sự chênh lệch này, nhưng khiến tất cả vector nhạc
cụ đều chỉ về cùng hướng → Cosine bị nén lại ở 0.999x.

**Giải pháp:** Dùng **Z-score Normalization** (chuẩn hóa từng chiều về mean=0, std=1 
trên toàn bộ dataset) + **L2 Distance** (khớp công thức Lp-norm trong Slide 8 môn học).

---

## Những thay đổi cần làm

### File 1: `feature_extraction.py`

#### 1a. Bỏ unit norm trong `extract_features()`

**Tìm và XÓA đoạn này** (khoảng dòng 55–63):

```python
# Chuan hoa thanh unit vector → tuong duong Cosine Similarity khi dung KD-Tree L2
norm = np.linalg.norm(features)
if norm > 0:
    features = features / norm
```

**Thay bằng** (chỉ return vector thô):

```python
features = [pitch, zcr, energy, centroid, bandwidth] + mfcc_mean
return features
```

#### 1b. Thêm Z-score trong `build_database()` trước khi build KD-Tree

**Tìm đoạn này** (gần cuối hàm `build_database()`):

```python
print(f"Build KD-Tree tu {len(songs)} windows...")
tree, _ = build_kdtree(songs)
with open(KDTREE_FILE, "wb") as f:
    pickle.dump({"tree": tree, "songs": songs}, f)
```

**Thay bằng:**

```python
print(f"Tinh Z-score va build KD-Tree tu {len(songs)} windows...")

vectors = np.array([s["features"] for s in songs], dtype=np.float32)
mean    = vectors.mean(axis=0)      # shape (18,)
std     = vectors.std(axis=0)       # shape (18,)
std[std == 0] = 1.0                  # tránh chia 0

vectors_norm = (vectors - mean) / std

tree = KDTree(vectors_norm)

with open(KDTREE_FILE, "wb") as f:
    pickle.dump({"tree": tree, "songs": songs, "mean": mean, "std": std}, f)
```

---

### File 2: `search.py`

#### 2a. Cập nhật `load_kdtree()` để trả về thêm `mean` và `std`

```python
# CŨ:
def load_kdtree():
    ...
    return data["tree"], data["songs"]

# MỚI:
def load_kdtree():
    if not os.path.exists(KDTREE_FILE):
        print(f"Chua co {KDTREE_FILE}. Hay chay: python main.py build")
        sys.exit(1)
    with open(KDTREE_FILE, "rb") as f:
        data = pickle.load(f)
    return data["tree"], data["songs"], data["mean"], data["std"]
```

#### 2b. Thay `l2_to_cosine()` bằng `l2_to_similarity()`

```python
# XÓA:
def l2_to_cosine(dist):
    return float(max(0.0, 1.0 - (dist ** 2) / 2.0))

# THÊM:
def l2_to_similarity(dist):
    """
    Chuyển L2 distance sang độ tương đồng trong khoảng (0, 1].
      dist = 0  → similarity = 1.0  (giống hoàn toàn)
      dist → ∞  → similarity → 0.0
    """
    return float(1.0 / (1.0 + dist))
```

#### 2c. Cập nhật hàm `search()` — normalize query và dùng metric mới

```python
# CŨ:
tree, songs = load_kdtree()
q_vec = np.array(extract_features(audio, sr), dtype=np.float32)
distances, idxs = tree.query(q_vec, k=k_query, p=2)

# MỚI:
tree, songs, mean, std = load_kdtree()
q_raw = np.array(extract_features(audio, sr), dtype=np.float32)
q_vec = (q_raw - mean) / std          # Z-score bằng cùng mean/std của DB
distances, idxs = tree.query(q_vec, k=k_query, p=2)
```

Thay tất cả `l2_to_cosine(dist)` → `l2_to_similarity(dist)`.

Cập nhật dòng mô tả metric:

```python
# CŨ:
print(f"DO DO         : Cosine Similarity (qua KD-Tree + unit norm)")

# MỚI:
print(f"DO DO         : L2 Distance (sau Z-score Normalization)")
```

---

### File 3: `evaluate.py`

Tìm và thay import:

```python
# CŨ:
from search import search, load_kdtree, load_query, extract_features, l2_to_cosine, TOP_K

# MỚI:
from search import search, load_kdtree, load_query, extract_features, l2_to_similarity, TOP_K
```

Thay tất cả `l2_to_cosine(dist)` → `l2_to_similarity(dist)` trong file này.

---

## Cách chạy sau khi sửa

> **Không cần xóa `music_database.db`** — dữ liệu thô trong SQLite giữ nguyên.

```bash
# Bước 1: Xóa KD-Tree cũ
del kdtree.pkl

# Bước 2: Build lại KD-Tree (chỉ mất vài giây, không đọc lại file audio)
python main.py build

# Bước 3: Test
python main.py search dataset/<tên_file>.wav
```

---

## Kết quả mong đợi

**Trước khi sửa:**
```
1      0.9999   Bai A    ← khó phân biệt
2      0.9998   Bai B
3      0.9997   Bai C
```

**Sau khi sửa:**
```
1      0.8542   Bai A    ← rất giống
2      0.4231   Bai B    ← khá giống
3      0.1823   Bai C    ← ít giống
```

Top 1 (chính file đó khi query file có trong DB) vẫn phải có similarity cao nhất.

---

## Ghi chú kỹ thuật

- Z-score mean/std tính trên **toàn bộ 8148 windows**, không phải từng file riêng lẻ
- `mean` và `std` được lưu trong `kdtree.pkl` để dùng khi normalize query
- Công thức: `sim = 1 / (1 + L2_distance)` — giá trị trong khoảng (0, 1]
- Khớp với công thức **Lp-norm (L2)** trong Slide 8 môn học
