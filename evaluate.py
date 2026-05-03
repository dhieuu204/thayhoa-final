import sys
import os
import time
import numpy as np
from feature_extraction import SR, WINDOW_SEC, get_song_name
from search import search, load_kdtree, load_query, extract_features, l2_to_cosine, TOP_K


# ─── Precision & Recall ───────────────────────────────────────────────────────

def precision_recall(results, relevant_names):
    """
    ca = Correct Accepted  (lay dung)
    fa = False Accepted    (lay sai)
    fd = False Dismissed   (bo sot)
    """
    retrieved     = [r["name"] for r in results]
    ca = sum(1 for r in retrieved if r in relevant_names)
    fa = len(retrieved) - ca
    fd = len(relevant_names) - ca
    precision = ca / (ca + fa) if (ca + fa) > 0 else 0.0
    recall    = ca / (ca + fd) if (ca + fd) > 0 else 0.0
    return {"ca": ca, "fa": fa, "fd": fd, "precision": precision, "recall": recall}


# ─── Benchmark: KD-Tree vs Brute-force ───────────────────────────────────────

def benchmark_search(query_vector, songs, runs=5):
    """So sanh toc do KD-Tree vs Brute-force L2 (noi dung Slide 8)."""
    from scipy.spatial import KDTree

    vectors = np.array([s["features"] for s in songs], dtype=np.float32)

    # --- KD-Tree ---
    tree = KDTree(vectors)
    t0   = time.time()
    for _ in range(runs):
        tree.query(query_vector, k=TOP_K, p=2)
    t_kd = (time.time() - t0) / runs

    # --- Brute-force L2 ---
    t0 = time.time()
    for _ in range(runs):
        dists = np.linalg.norm(vectors - query_vector, axis=1)
        np.argsort(dists)[:TOP_K]
    t_bf = (time.time() - t0) / runs

    return t_kd, t_bf


# ─── Demo Case 1: file CO trong DB ───────────────────────────────────────────

def demo_in_db(query_path, relevant_names=None):
    print("\n" + "=" * 55)
    print("  DEMO CASE 1: FILE DA CO TRONG DB")
    print("=" * 55)

    results    = search(query_path)
    query_name = get_song_name(query_path)

    # Kiem tra Top 1 co phai chinh no khong
    top1 = results[0]
    print(f"\n[KIEM TRA] Top 1 co phai chinh no khong?")
    if top1["distance"] < 1e-4:
        print(f"  DUNG — '{top1['name']}' voi khoang cach = {top1['distance']:.6f}")
    else:
        print(f"  SAI  — Top 1 la '{top1['name']}' (distance={top1['distance']:.4f})")

    if relevant_names is None:
        relevant_names = [query_name]

    metrics = precision_recall(results, relevant_names)
    print(f"\n[DANH GIA]")
    print(f"  Correct Accepted (ca) : {metrics['ca']}")
    print(f"  False Accepted   (fa) : {metrics['fa']}")
    print(f"  False Dismissed  (fd) : {metrics['fd']}")
    print(f"  Precision             : {metrics['precision']:.2%}")
    print(f"  Recall                : {metrics['recall']:.2%}")

    return results, metrics


# ─── Demo Case 2: file NGOAI DB ──────────────────────────────────────────────

def demo_out_db(query_path, relevant_names=None):
    print("\n" + "=" * 55)
    print("  DEMO CASE 2: FILE CHUA CO TRONG DB")
    print("=" * 55)

    results = search(query_path)
    top1    = results[0]
    print(f"\n[KIEM TRA] Bai gan nhat: '{top1['name']}' (cosine={top1['similarity']:.4f})")

    if relevant_names:
        metrics = precision_recall(results, relevant_names)
        print(f"\n[DANH GIA]")
        print(f"  Correct Accepted (ca) : {metrics['ca']}")
        print(f"  False Accepted   (fa) : {metrics['fa']}")
        print(f"  False Dismissed  (fd) : {metrics['fd']}")
        print(f"  Precision             : {metrics['precision']:.2%}")
        print(f"  Recall                : {metrics['recall']:.2%}")
    else:
        print("  (Khong co danh sach relevant → bo qua Precision/Recall)")
        metrics = None

    return results, metrics


# ─── Benchmark toc do ────────────────────────────────────────────────────────

def demo_benchmark(query_path):
    print("\n" + "=" * 55)
    print("  BENCHMARK: KD-Tree vs Brute-force L2")
    print("  (Noi dung Slide 8 — Truy van khong gian vector)")
    print("=" * 55)

    print("\n  Trich xuat vector query...")
    audio, sr    = load_query(query_path)
    query_vector = np.array(extract_features(audio, sr), dtype=np.float32)

    _, songs = load_kdtree()
    print(f"  So windows trong DB : {len(songs)}")
    print(f"  Chay benchmark (5 lan / phuong phap)...\n")

    t_kd, t_bf = benchmark_search(query_vector, songs, runs=5)

    speedup = t_bf / t_kd if t_kd > 0 else float("inf")

    print(f"  {'Phuong phap':<25} {'Thoi gian TB':>14}  {'Do phuc tap'}")
    print(f"  {'-'*60}")
    print(f"  {'KD-Tree':<25} {t_kd*1000:>11.3f} ms  O(D x log N)")
    print(f"  {'Brute-force L2':<25} {t_bf*1000:>11.3f} ms  O(N x D)")
    print(f"\n  KD-Tree nhanh hon Brute-force: {speedup:.1f}x")
    print("=" * 55)


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
        print("  python evaluate.py out  <file.wav> [ten1,ten2,...]")
        print("  python evaluate.py bench <file.wav>")
