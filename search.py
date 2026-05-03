import sys
import os
import time
import pickle
import numpy as np
import librosa
from feature_extraction import extract_features, SR, WINDOW_SEC, DB_FILE, KDTREE_FILE

TOP_K   = 5
MIN_SEC = 3.0   # file query toi thieu 3 giay


# ─── Load KD-Tree ─────────────────────────────────────────────────────────────

def load_kdtree():
    if not os.path.exists(KDTREE_FILE):
        print(f"Chua co {KDTREE_FILE}. Hay chay: python main.py build")
        sys.exit(1)
    with open(KDTREE_FILE, "rb") as f:
        data = pickle.load(f)
    return data["tree"], data["songs"]


# ─── Chuan bi query vector ────────────────────────────────────────────────────

def load_query(query_path):
    """
    Doc file query, cat lay doan WINDOW_SEC giay dau tien.
    Neu ngan hon WINDOW_SEC thi tile (pad) len du do dai.
    """
    duration = librosa.get_duration(path=query_path)
    if duration < MIN_SEC:
        print(f"Query qua ngan ({duration:.1f}s). Can it nhat {MIN_SEC}s.")
        sys.exit(1)

    audio, sr  = librosa.load(query_path, sr=SR, mono=True)
    target_len = int(WINDOW_SEC * sr)

    if len(audio) < target_len:
        # Pad bang cach lap lai am thanh
        audio = np.tile(audio, int(np.ceil(target_len / len(audio))))

    return audio[:target_len], sr


# ─── Tinh do tuong dong Cosine tu L2 cua unit vector ─────────────────────────

def l2_to_cosine(dist):
    """
    Cong thuc chinh xac: neu q va d deu la unit vector,
    thi  ||q - d||^2 = 2 - 2*cos(theta)
    => cos(theta) = 1 - dist^2 / 2
    """
    return float(max(0.0, 1.0 - (dist ** 2) / 2.0))


# ─── Ham search chinh ─────────────────────────────────────────────────────────

def search(query_path):
    print("=" * 55)
    print(f"FILE TRUY VAN : {os.path.basename(query_path)}")
    print(f"DO DO         : Cosine Similarity (qua KD-Tree + unit norm)")
    print("=" * 55)

    # Buoc 1: Trich xuat vector query
    print("\n[1] Trich xuat dac trung file truy van...")
    t0          = time.time()
    audio, sr   = load_query(query_path)
    q_vec       = np.array(extract_features(audio, sr), dtype=np.float32)
    t_extract   = time.time() - t0

    labels = ["Pitch(Hz)", "ZCR", "Energy", "Centroid", "Bandwidth"] + \
             [f"MFCC_{i}" for i in range(1, 14)]
    print(f"    Thoi gian trich xuat : {t_extract:.2f}s")
    print(f"    Vector {len(q_vec)} chieu (da chuan hoa unit norm):")
    for label, val in zip(labels[:5], q_vec[:5]):
        print(f"      {label:12s} = {val:.6f}")
    print(f"      MFCC_1..13   = [{', '.join(f'{v:.3f}' for v in q_vec[5:])}]")

    # Buoc 2: Tim kiem bang KD-Tree
    print(f"\n[2] Tim kiem {TOP_K} lang gieng gan nhat (KD-Tree, Cosine)...")
    tree, songs = load_kdtree()

    t0 = time.time()
    # KD-Tree dung khoang cach L2 tren unit vector → tuong duong Cosine
    k_query          = min(TOP_K * 10, len(songs))   # lay nhieu hon de dedup
    distances, idxs  = tree.query(q_vec, k=k_query, p=2)
    t_search         = time.time() - t0

    print(f"    So windows trong DB  : {len(songs)}")
    print(f"    Thoi gian tim kiem   : {t_search * 1000:.3f}ms")

    # Buoc 3: Ket qua trung gian — top windows tim duoc
    print(f"\n[3] Ket qua trung gian (top windows):")
    print(f"    {'STT':<5} {'Cosine':>8}  {'Offset':>7}  Ten bai")
    print(f"    {'-'*55}")
    for rank, (dist, idx) in enumerate(zip(distances[:15], idxs[:15]), 1):
        s   = songs[idx]
        cos = l2_to_cosine(dist)
        marker = " <- CHINH NO" if dist < 1e-4 else ""
        print(f"    {rank:<5} {cos:>7.4f}  {s['offset']:>5.1f}s  {s['name'][:35]}{marker}")

    # Buoc 4: Dedup — giu moi bai 1 lan (window co cosine cao nhat)
    seen = {}
    for dist, idx in zip(distances, idxs):
        s   = songs[idx]
        cos = l2_to_cosine(dist)
        if s["name"] not in seen or cos > seen[s["name"]]["similarity"]:
            seen[s["name"]] = {**s, "distance": float(dist), "similarity": cos}

    top5 = sorted(seen.values(), key=lambda x: -x["similarity"])[:TOP_K]

    # Buoc 5: Hien thi ket qua cuoi
    print(f"\n[4] KET QUA TOP {TOP_K} BAI NHAC TUONG DONG:")
    print(f"    {'Hang':<6} {'Cosine':>8}  {'Tuong dong':>12}  Ten bai")
    print(f"    {'-'*60}")
    for rank, r in enumerate(top5, 1):
        pct = r["similarity"] * 100
        print(f"    {rank:<6} {r['similarity']:>8.4f}  {pct:>11.2f}%  {r['name']}")
        print(f"           File: {r['path']}")

    print("=" * 55)
    return top5


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cach dung: python search.py <file_audio.wav>")
        print("Vi du    : python search.py query.wav")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Khong tim thay file: {sys.argv[1]}")
        sys.exit(1)
    search(sys.argv[1])
