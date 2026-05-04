import os
import glob
import sys
import time
import json
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.spatial import KDTree
from math import gcd
import pymysql

# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR = "dataset"
KDTREE_FILE = "kdtree.pkl"
NORMS_FILE  = "norms.json"
SR_TARGET   = 22050
WINDOW_SEC  = 5.0
STEP_SEC    = 2.5
N_WORKERS   = 8
N_DIM       = 6

DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "kali",
    "database": "MusicDB",
    "charset":  "utf8mb4",
}


# ─── MySQL ────────────────────────────────────────────────────────────────────

def get_conn():
    return pymysql.connect(**DB_CONFIG)


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id            INT AUTO_INCREMENT PRIMARY KEY,
                song_name     VARCHAR(500)  NOT NULL,
                file_path     VARCHAR(1000) NOT NULL,
                offset_sec    DOUBLE        NOT NULL,
                pitch         DOUBLE,
                zcr           DOUBLE,
                energy        DOUBLE,
                centroid      DOUBLE,
                bandwidth     DOUBLE,
                harmonicity   DOUBLE,
                z_pitch       DOUBLE,
                z_zcr         DOUBLE,
                z_energy      DOUBLE,
                z_centroid    DOUBLE,
                z_bandwidth   DOUBLE,
                z_harmonicity DOUBLE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
    conn.commit()


def get_existing_paths(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT file_path FROM frames")
        rows = cur.fetchall()
    return set(r[0] for r in rows)


def insert_batch(conn, rows):
    """rows: list of (song_name, file_path, offset_sec, [pitch,zcr,energy,centroid,bw,harm])"""
    data = [
        (name, fp, offset,
         feat[0], feat[1], feat[2], feat[3], feat[4], feat[5],
         None, None, None, None, None, None)
        for name, fp, offset, feat in rows
    ]
    with conn.cursor() as cursor:
        cursor.executemany("""
            INSERT INTO frames
                (song_name, file_path, offset_sec,
                 pitch, zcr, energy, centroid, bandwidth, harmonicity,
                 z_pitch, z_zcr, z_energy, z_centroid, z_bandwidth, z_harmonicity)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, data)
    conn.commit()


def load_all_raw(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, song_name, file_path, offset_sec,
                   pitch, zcr, energy, centroid, bandwidth, harmonicity
            FROM frames ORDER BY id
        """)
        return cur.fetchall()


def update_zscores(conn, updates):
    """updates: list of (z_pitch, z_zcr, z_energy, z_centroid, z_bw, z_harm, id)"""
    with conn.cursor() as cursor:
        cursor.executemany("""
            UPDATE frames SET
                z_pitch=%s, z_zcr=%s, z_energy=%s,
                z_centroid=%s, z_bandwidth=%s, z_harmonicity=%s
            WHERE id=%s
        """, updates)
    conn.commit()


def load_zscores(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, song_name, file_path, offset_sec,
                   z_pitch, z_zcr, z_energy, z_centroid, z_bandwidth, z_harmonicity
            FROM frames ORDER BY id
        """)
        rows = cur.fetchall()
    return [
        {
            "id": r[0], "name": r[1], "path": r[2], "offset": r[3],
            "features": np.array([r[4], r[5], r[6], r[7], r[8], r[9]], dtype=np.float32)
        }
        for r in rows
    ]


# ─── Tên bài ──────────────────────────────────────────────────────────────────

def get_song_name(file_path):
    import re
    stem = os.path.splitext(os.path.basename(file_path))[0]
    name = re.sub(r'^IRMAS-[^_]+__[^_]+__', '', stem).strip()
    return name if name else stem


def get_base_song_name(file_path):
    """Bỏ phần số segment đuôi: 'Avocet-13.wav' → 'Avocet'."""
    import re
    name = get_song_name(file_path)
    return re.sub(r'-\d+$', '', name)


# ─── Load audio ───────────────────────────────────────────────────────────────

def load_audio(file_path, sr_target=SR_TARGET):
    """Đọc WAV bằng scipy, convert mono float32, resample nếu cần."""
    sr_orig, data = wavfile.read(file_path)

    # Convert sang float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample về SR_TARGET nếu khác
    if sr_orig != sr_target:
        g = gcd(int(sr_orig), int(sr_target))
        data = resample_poly(data, sr_target // g, sr_orig // g)

    return data, sr_target


# ─── Công thức trích xuất đặc trưng (lấy từ slide tài liệu môn học) ──────────

def feat_energy(x):
    """Năng lượng trung bình — Slide 10, trang 7
    E = (1/N) * Σ x(n)²
    """
    N = len(x)
    return float(np.sum(x * x) / N)


def feat_zcr(x):
    """Tốc độ đổi dấu — Slide 10, trang 7
    ZC = Σ |sgn(x[n]) - sgn(x[n-1])| / (2N)
    sgn(a) = 1 nếu a>0, 0 nếu a=0, -1 nếu a<0
    """
    N = len(x)
    sgn = np.sign(x)
    zc = float(np.sum(np.abs(sgn[1:] - sgn[:-1])) / (2 * N))
    return zc


def _dft_magnitude(x, sr):
    """Phổ biên độ DFT — Slide 10, trang 10
    X[k] = Σ x(n) * e^(-j2πnk/N)
    Dùng rfft (tín hiệu thực) để lấy nửa phổ dương.
    """
    N = len(x)
    X = np.abs(np.fft.rfft(x))           # biên độ phổ
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)  # tần số tương ứng (Hz)
    return X, freqs


def feat_centroid(x, sr):
    """Trọng tâm phổ (brightness) — Slide 10, trang 12
    Trọng tâm = Σ(f_k * |X[k]|) / Σ|X[k]|
    """
    X, freqs = _dft_magnitude(x, sr)
    denom = np.sum(X)
    if denom < 1e-10:
        return 0.0
    return float(np.sum(freqs * X) / denom)


def feat_bandwidth(x, sr):
    """Băng thông — Slide 10, trang 12
    BW = f_cao - f_thap  (các thành phần phổ 'dương': ≥ threshold)
    Threshold: 3dB trên mức sàn nhiễu → threshold = min_val * √2
    """
    X, freqs = _dft_magnitude(x, sr)
    if np.max(X) < 1e-10:
        return 0.0
    positive = X[X > 0]
    if len(positive) == 0:
        return 0.0
    noise_floor = np.min(positive)
    threshold = noise_floor * np.sqrt(2)   # +3dB so với mức câm
    mask = X >= threshold
    above = freqs[mask]
    if len(above) < 2:
        return 0.0
    return float(above[-1] - above[0])


def _autocorrelation_fft(x):
    """Tự tương quan chuẩn hóa dùng FFT — O(N log N)
    R[τ] = Σ x[n] * x[n+τ]  (dùng định lý Wiener-Khinchin)
    """
    N = len(x)
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    power = X * np.conj(X)                    # |X[k]|²
    r = np.real(np.fft.irfft(power, n=n_fft))[:N]
    denom = r[0]
    return r / (denom + 1e-10)


def feat_pitch(x, sr):
    """Tần số cơ bản (Pitch / F0) — Slide 10, trang 14
    Ước lượng F0 bằng đỉnh tự tương quan:
      Lag tương ứng với chu kỳ → F0 = sr / lag
    Khoảng tìm: 50 Hz – 500 Hz.
    """
    r = _autocorrelation_fft(x)
    min_lag = max(1, int(sr / 500))   # 500 Hz tối đa
    max_lag = min(int(sr / 50), len(r) - 1)   # 50 Hz tối thiểu
    if min_lag >= max_lag:
        return 0.0
    peak_lag = int(np.argmax(r[min_lag:max_lag])) + min_lag
    if r[peak_lag] < 0.3:   # tín hiệu không đủ tính tuần hoàn
        return 0.0
    return float(sr / peak_lag)


def feat_harmonicity(x, sr):
    """Độ điều hòa âm — Slide 10, trang 13
    Âm nhạc có tính điều hòa: các thành phần phổ là bội số của F0.
    Đo bằng giá trị đỉnh tự tương quan tại lag > 0 (= tỉ lệ năng lượng điều hòa).
    """
    r = _autocorrelation_fft(x)
    min_lag = max(1, int(sr / 500))
    max_lag = min(int(sr / 50), len(r) - 1)
    if min_lag >= max_lag:
        return 0.0
    return float(max(0.0, np.max(r[min_lag:max_lag])))


def extract_features(x, sr):
    """Trích xuất vector 6 chiều: [pitch, zcr, energy, centroid, bandwidth, harmonicity]"""
    return [
        feat_pitch(x, sr),
        feat_zcr(x),
        feat_energy(x),
        feat_centroid(x, sr),
        feat_bandwidth(x, sr),
        feat_harmonicity(x, sr),
    ]


# ─── Sliding window ───────────────────────────────────────────────────────────

def process_file(file_path):
    """Cắt file thành frame 5s/2.5s, trích xuất đặc trưng mỗi frame."""
    name = get_song_name(file_path)
    audio, sr = load_audio(file_path)
    window_len = int(WINDOW_SEC * sr)
    step_len   = int(STEP_SEC   * sr)
    results = []
    for start in range(0, len(audio) - window_len + 1, step_len):
        frame = audio[start: start + window_len]
        feats = extract_features(frame, sr)
        results.append((name, file_path, start / sr, feats))
    return results


# ─── Z-score chuẩn hóa ───────────────────────────────────────────────────────

def compute_zscore_params(all_features):
    """Tính μ và σ trên toàn bộ frame — kien.md phần 4
    μ_j = (1/M) * Σ v_ij
    σ_j = sqrt( (1/M) * Σ (v_ij - μ_j)² )
    M: tổng số frame, j: chiều thứ j (j=1..6)
    """
    mat   = np.array(all_features, dtype=np.float64)   # (M, 6)
    mu    = mat.mean(axis=0)                            # (6,)
    sigma = np.sqrt(((mat - mu) ** 2).mean(axis=0))    # (6,)
    sigma[sigma < 1e-10] = 1.0                         # tránh chia 0
    return mu.tolist(), sigma.tolist()


def zscore(v, mu, sigma):
    """z_ij = (v_ij - μ_j) / σ_j — kien.md phần 4"""
    return [(v[j] - mu[j]) / sigma[j] for j in range(len(v))]


# ─── Build database ───────────────────────────────────────────────────────────

def build_database(reset=False):
    conn = get_conn()

    if reset:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS frames")
        conn.commit()
        for f in [KDTREE_FILE, NORMS_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Da xoa: {f}")

    init_db(conn)
    existing = get_existing_paths(conn)

    wav_files = sorted(set(
        f for f in
        glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
        + glob.glob(os.path.join(DATASET_DIR, "*.wav"))
        if not os.path.basename(f).startswith("_temp")
    ))

    if not wav_files:
        print(f"Khong tim thay file WAV trong '{DATASET_DIR}/'")
        conn.close()
        return

    todo    = [f for f in wav_files if f not in existing]
    total   = len(wav_files)
    skipped = total - len(todo)
    print(f"Tim thay {total} file WAV | Bo qua: {skipped} | Can xu ly: {len(todo)}")
    print(f"Window: {WINDOW_SEC}s | Step: {STEP_SEC}s | Workers: {N_WORKERS}\n")

    if todo:
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_file, fp): fp for fp in todo}
            done = 0
            for future in as_completed(futures):
                fp = futures[future]
                done += 1
                try:
                    rows = future.result()
                    insert_batch(conn, rows)
                    print(f"  [{done:4}/{len(todo)}] {os.path.basename(fp)[:50]:50s} | {len(rows)} frames")
                except Exception as e:
                    print(f"  [{done:4}/{len(todo)}] {os.path.basename(fp)}: LOI - {e}")
        print(f"\nHoan thanh trich xuat ({time.time()-t_start:.1f}s)\n")

    # ── Z-score toàn bộ frame ──
    print("Tinh Z-score toan bo frame...")
    raw_rows = load_all_raw(conn)
    if not raw_rows:
        print("DB trong.")
        conn.close()
        return

    all_feats = [[r[4], r[5], r[6], r[7], r[8], r[9]] for r in raw_rows]
    mu, sigma = compute_zscore_params(all_feats)

    with open(NORMS_FILE, "w") as f:
        json.dump({"mu": mu, "sigma": sigma}, f, indent=2)
    print(f"Da luu mu, sigma -> {NORMS_FILE}")

    updates = []
    for r, raw in zip(raw_rows, all_feats):
        z = zscore(raw, mu, sigma)
        updates.append((z[0], z[1], z[2], z[3], z[4], z[5], r[0]))
    update_zscores(conn, updates)
    print(f"Da cap nhat z-score cho {len(updates)} frame vao DB")

    # ── Build KD-Tree ──
    songs = load_zscores(conn)
    conn.close()

    print(f"\nBuild KD-Tree tu {len(songs)} frame...")
    vectors = np.array([s["features"] for s in songs], dtype=np.float32)
    tree = KDTree(vectors)
    with open(KDTREE_FILE, "wb") as f:
        pickle.dump({"tree": tree, "songs": songs}, f)

    print(f"\n{'='*55}")
    print(f"Tong bai trong DB  : {len(set(s['name'] for s in songs))}")
    print(f"Tong frame         : {len(songs)}")
    print(f"So chieu vector    : {N_DIM}")
    print(f"KD-Tree            : {KDTREE_FILE}")
    print(f"Norms (mu/sigma)   : {NORMS_FILE}")
    print(f"MySQL table        : frames")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    build_database(reset=reset)
