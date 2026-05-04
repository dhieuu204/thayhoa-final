import sys
import os
import time
import json
import pickle
import numpy as np
from feature_extraction import (
    load_audio, extract_features, zscore,
    get_song_name, SR_TARGET, WINDOW_SEC, STEP_SEC,
    KDTREE_FILE, NORMS_FILE
)

TOP_K   = 5
MIN_SEC = 3.0
K_CAND  = 10   # số frame ứng viên mỗi query frame — kien.md bước 3


# ─── Load KD-Tree & norms ─────────────────────────────────────────────────────

def load_kdtree():
    if not os.path.exists(KDTREE_FILE):
        print(f"Chua co {KDTREE_FILE}. Hay chay: python main.py build")
        sys.exit(1)
    with open(KDTREE_FILE, "rb") as f:
        data = pickle.load(f)
    return data["tree"], data["songs"]


def load_norms():
    if not os.path.exists(NORMS_FILE):
        print(f"Chua co {NORMS_FILE}. Hay chay: python main.py build")
        sys.exit(1)
    with open(NORMS_FILE) as f:
        d = json.load(f)
    return d["mu"], d["sigma"]


# ─── Đo khoảng cách & độ tương đồng ──────────────────────────────────────────

def l2_distance(a, b):
    """L2-norm — Slide 8, trang 17
    L2(Di, Dj) = sqrt( Σ (z_ik - z_jk)² )
    """
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))


def similarity(dist):
    """Độ tương đồng — kien.md phần 6
    sim = 1 / (1 + L2)
    sim = 1  → giống hoàn toàn
    sim → 0  → càng khác nhau
    """
    return 1.0 / (1.0 + dist)


# ─── Hàm search chính ────────────────────────────────────────────────────────

def search(query_path):
    print("=" * 60)
    print(f"FILE TRUY VAN : {os.path.basename(query_path)}")
    print(f"DO DO         : sim = 1/(1+L2)  |  Chuan hoa: Z-score")
    print("=" * 60)

    mu, sigma   = load_norms()
    tree, songs = load_kdtree()

    # ── Bước 1: cửa sổ trượt file query ──────────────────────────────────────
    audio, sr = load_audio(query_path)
    duration  = len(audio) / sr
    if duration < MIN_SEC:
        print(f"File qua ngan ({duration:.1f}s). Can it nhat {MIN_SEC}s.")
        sys.exit(1)

    window_len = int(WINDOW_SEC * sr)
    step_len   = int(STEP_SEC   * sr)

    # Pad nếu ngắn hơn 1 window (lặp lại tín hiệu)
    if len(audio) < window_len:
        repeats = int(np.ceil(window_len / len(audio)))
        audio   = np.tile(audio, repeats)

    q_frames = []
    for start in range(0, len(audio) - window_len + 1, step_len):
        q_frames.append(audio[start: start + window_len])
    if not q_frames:
        q_frames = [audio[:window_len]]

    Q = len(q_frames)
    print(f"\n[1] File query -> {Q} frame  (window={WINDOW_SEC}s, step={STEP_SEC}s)")

    # ── Bước 2 & 3: mỗi frame → trích xuất + Z-score → KD-tree ──────────────
    labels = ["Pitch", "ZCR", "Energy", "Centroid", "Bandwidth", "Harmony"]
    print(f"\n[2] Trich xuat + Z-score + KD-tree ({K_CAND} ung vien/frame)...")

    t0    = time.time()
    score = {}   # song_name → tổng sim tích lũy
    best  = {}   # song_name → thông tin frame match tốt nhất

    for i, frame in enumerate(q_frames):
        # Bước 2: trích xuất 6 thuộc tính
        raw  = extract_features(frame, sr)
        # Z-score dùng mu/sigma đã lưu từ lúc build
        zvec = np.array(zscore(raw, mu, sigma), dtype=np.float32)

        if i == 0:
            print(f"\n    Frame 0 — vector raw  : {[f'{v:.4f}' for v in raw]}")
            print(f"    Frame 0 — vector Z    : {[f'{v:.4f}' for v in zvec]}")
            print(f"    {'Label':<12}: {'raw':>10}  {'z-score':>10}")
            print(f"    {'-'*36}")
            for lbl, rv, zv in zip(labels, raw, zvec):
                print(f"    {lbl:<12}: {rv:>10.4f}  {zv:>10.4f}")

        # Bước 3: KD-tree tìm K_CAND frame gần nhất
        k_query = min(K_CAND, len(songs))
        dists, idxs = tree.query(zvec, k=k_query, p=2)

        # Dedup: mỗi bài chỉ lấy frame tốt nhất trong query frame này
        best_this_frame = {}
        for dist, idx in zip(dists, idxs):
            s    = songs[idx]
            name = s["name"]
            sim  = similarity(dist)
            if name not in best_this_frame or sim > best_this_frame[name]["sim"]:
                best_this_frame[name] = {"s": s, "sim": sim, "dist": dist}

        # Bước 4: cộng dồn điểm (mỗi bài tối đa 1 lần / query frame)
        for name, info in best_this_frame.items():
            score[name] = score.get(name, 0.0) + info["sim"]
            if name not in best or info["sim"] > best[name]["similarity"]:
                best[name] = {**info["s"], "distance": float(info["dist"]), "similarity": info["sim"]}

    t_search = time.time() - t0

    # ── Bước 5: chuẩn hóa điểm theo số frame query ───────────────────────────
    for name in score:
        score[name] /= Q

    print(f"\n    Tong frame query (Q)  : {Q}")
    print(f"    So bai ung vien       : {len(score)}")
    print(f"    Thoi gian tim kiem    : {t_search*1000:.2f}ms")
    print(f"    So windows trong DB   : {len(songs)}")

    # ── Bước 6: sắp xếp → Top 5 ──────────────────────────────────────────────
    sorted_cands = sorted(score.items(), key=lambda x: -x[1])

    print(f"\n[3] Ket qua trung gian (score gom tu {Q} frame, chuan hoa /Q):")
    print(f"    {'STT':<5} {'Score':>8}  Ten bai")
    print(f"    {'-'*52}")
    for rank, (name, sc) in enumerate(sorted_cands[:10], 1):
        print(f"    {rank:<5} {sc:>8.4f}  {name}")

    top5 = []
    for name, sc in sorted_cands[:TOP_K]:
        top5.append({**best[name], "score": sc})

    print(f"\n[4] KET QUA TOP {TOP_K} BAI NHAC TUONG DONG:")
    print(f"    {'Hang':<6} {'Score':>8}  {'%':>8}  Ten bai")
    print(f"    {'-'*62}")
    for rank, r in enumerate(top5, 1):
        pct = r["score"] * 100
        print(f"    {rank:<6} {r['score']:>8.4f}  {pct:>7.2f}%  {r['name']}")
        print(f"           File: {r['path']}")

    print("=" * 60)
    return top5


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cach dung: python search.py <file_audio.wav>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Khong tim thay file: {sys.argv[1]}")
        sys.exit(1)
    search(sys.argv[1])
