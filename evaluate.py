import sys
import os
import time
import numpy as np
from feature_extraction import SR_TARGET, WINDOW_SEC, get_base_song_name
from search import search, load_kdtree, load_norms, load_audio, extract_features, zscore, similarity, TOP_K, K_CAND, count_segments_in_db


# ─── Precision & Recall ───────────────────────────────────────────────────────

def precision_recall(results, relevant_names):
    """
    ca = Correct Accepted  (lay dung)
    fa = False Accepted    (lay sai)
    fd = False Dismissed   (dung nhung bo sot)
    """
    retrieved = [r["path"] for r in results]
    ca = sum(1 for r in retrieved if r in relevant_names)
    fa = len(retrieved) - ca
    fd = len(relevant_names) - ca
    precision = ca / (ca + fa) if (ca + fa) > 0 else 0.0
    recall    = ca / (ca + fd) if (ca + fd) > 0 else 0.0
    return {"ca": ca, "fa": fa, "fd": fd, "precision": precision, "recall": recall}


# ─── Benchmark: KD-Tree vs Brute-force ───────────────────────────────────────

def benchmark_search(query_vector, songs, runs=5):
    """So sanh toc do KD-Tree vs Brute-force L2 — noi dung Slide 8."""
    from scipy.spatial import KDTree

    vectors = np.array([s["features"] for s in songs], dtype=np.float32)
    tree    = KDTree(vectors)

    # KD-Tree
    t0 = time.time()
    for _ in range(runs):
        tree.query(query_vector, k=TOP_K, p=2)
    t_kd = (time.time() - t0) / runs

    # Brute-force L2 — Slide 8, trang 17: L2 = sqrt(Σ(z_ik - z_jk)²)
    t0 = time.time()
    for _ in range(runs):
        diffs = vectors - query_vector
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        np.argsort(dists)[:TOP_K]
    t_bf = (time.time() - t0) / runs

    return t_kd, t_bf


# ─── Demo Case 1: file CÓ trong DB ───────────────────────────────────────────

def demo_in_db(query_path, relevant_names=None):
    print("\n" + "=" * 60)
    print("  DEMO CASE 1: FILE DA CO TRONG DB")
    print("=" * 60)

    # ── K động: đếm segment của bài này trong DB ──────────────────────────────
    _, songs  = load_kdtree()
    base_name = get_base_song_name(query_path)
    dyn_k     = count_segments_in_db(base_name, songs)
    if dyn_k == 0:
        dyn_k = TOP_K
        print(f"  (Khong tim thay bai '{base_name}' trong DB, dung K mac dinh={dyn_k})")

    # Lấy đủ kết quả cho cả Precision@5 và Recall@K
    fetch_k = max(5, dyn_k)
    results = search(query_path, top_k=fetch_k)

    top1 = results[0]
    print(f"\n[KIEM TRA] Top 1 co phai chinh no khong?")
    if top1["distance"] < 1e-4:
        print(f"  DUNG — '{top1['path']}' voi khoang cach = {top1['distance']:.6f}")
    else:
        print(f"  SAI  — Top 1 la '{top1['path']}' (distance={top1['distance']:.4f})")

    if relevant_names is None:
        relevant_names = [query_path]

    metrics = precision_recall(results[:5], relevant_names)
    print(f"\n[DANH GIA]")
    print(f"  Correct Accepted (ca) : {metrics['ca']}")
    print(f"  False Accepted   (fa) : {metrics['fa']}")
    print(f"  False Dismissed  (fd) : {metrics['fd']}")
    print(f"  Precision             : {metrics['precision']:.2%}")
    print(f"  Recall                : {metrics['recall']:.2%}")

    # ── Precision@5: P = ca / (ca + fa) ──────────────────────────────────────
    ca_5  = sum(1 for r in results[:5] if get_base_song_name(r["path"]) == base_name)
    fa_5  = 5 - ca_5          # lay sai: khong dung bai nhung he thong chon
    p_5   = ca_5 / (ca_5 + fa_5) if (ca_5 + fa_5) > 0 else 0.0

    # ── Recall@K:  R = ca / (ca + fd) ─────────────────────────────────────────
    ca_k  = sum(1 for r in results[:dyn_k] if get_base_song_name(r["path"]) == base_name)
    fd_k  = dyn_k - ca_k      # loai bo sai: dung bai nhung he thong bo qua
    r_k   = ca_k / (ca_k + fd_k) if (ca_k + fd_k) > 0 else 0.0

    print(f"\n[DANH GIA SONG-LEVEL]")
    print(f"  Bai goc (query)       : {base_name}")
    print(f"  K dong                : {dyn_k}  (tong segment cua bai trong DB)")
    print(f"  --- Precision@5 (top 5 co dang tin khong?) ---")
    print(f"  ca (chon dung)        : {ca_5}")
    print(f"  fa (chon sai)         : {fa_5}")
    print(f"  Precision@5           : {ca_5}/{ca_5+fa_5} = {p_5:.2%}")
    print(f"  --- Recall@{dyn_k} (tim duoc bao nhieu segment?) ---")
    print(f"  ca (tim duoc dung)    : {ca_k}")
    print(f"  fd (bo sot)           : {fd_k}")
    print(f"  Recall@{dyn_k:<5}          : {ca_k}/{ca_k+fd_k} = {r_k:.2%}")

    return results, metrics


# ─── Demo Case 2: file NGOÀI DB ──────────────────────────────────────────────

def demo_out_db(query_path, relevant_names=None):
    print("\n" + "=" * 60)
    print("  DEMO CASE 2: FILE CHUA CO TRONG DB")
    print("=" * 60)

    results = search(query_path)
    top1    = results[0]
    print(f"\n[KIEM TRA] File gan nhat: '{top1['path']}' (score={top1['score']:.4f})")

    if relevant_names:
        metrics = precision_recall(results, relevant_names)
        print(f"\n[DANH GIA]")
        print(f"  Correct Accepted (ca) : {metrics['ca']}")
        print(f"  False Accepted   (fa) : {metrics['fa']}")
        print(f"  False Dismissed  (fd) : {metrics['fd']}")
        print(f"  Precision             : {metrics['precision']:.2%}")
        print(f"  Recall                : {metrics['recall']:.2%}")
    else:
        print("  (Khong co danh sach relevant -> bo qua Precision/Recall)")
        metrics = None

    return results, metrics


# ─── Benchmark tốc độ ────────────────────────────────────────────────────────

def demo_benchmark(query_path):
    print("\n" + "=" * 60)
    print("  BENCHMARK: KD-Tree vs Brute-force L2")
    print("  (Noi dung Slide 8 — Truy van khong gian vector)")
    print("=" * 60)

    mu, sigma    = load_norms()
    audio, sr    = load_audio(query_path)
    window_len   = int(WINDOW_SEC * sr)
    if len(audio) < window_len:
        audio = np.tile(audio, int(np.ceil(window_len / len(audio))))
    frame        = audio[:window_len]
    raw          = extract_features(frame, sr)
    query_vector = np.array(zscore(raw, mu, sigma), dtype=np.float32)

    _, songs = load_kdtree()
    print(f"\n  So frame trong DB   : {len(songs)}")
    print(f"  Chay benchmark (5 lan / phuong phap)...\n")

    t_kd, t_bf = benchmark_search(query_vector, songs, runs=5)
    speedup = t_bf / t_kd if t_kd > 0 else float("inf")

    print(f"  {'Phuong phap':<25} {'Thoi gian TB':>14}  {'Do phuc tap'}")
    print(f"  {'-'*60}")
    print(f"  {'KD-Tree':<25} {t_kd*1000:>11.3f} ms  O(D x log N)")
    print(f"  {'Brute-force L2':<25} {t_bf*1000:>11.3f} ms  O(N x D)")
    print(f"\n  KD-Tree nhanh hon Brute-force: {speedup:.1f}x")
    print("=" * 60)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        mode = sys.argv[1]
        path = sys.argv[2]
        if mode == "in":
            demo_in_db(path)
        elif mode == "out":
            relevant = sys.argv[3].split(",") if len(sys.argv) > 3 else None
            demo_out_db(path, relevant_names=relevant)
        elif mode == "bench":
            demo_benchmark(path)
    else:
        print("Cach dung:")
        print("  python evaluate.py in   <file.wav>")
        print("  python evaluate.py out  <file.wav> [path1,path2,...]")
        print("  python evaluate.py bench <file.wav>")
