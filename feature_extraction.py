import os
import glob
import sys
import time
import pickle
import sqlite3
import multiprocessing
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree

DATASET_DIR = "dataset"
DB_FILE     = "music_database.db"
KDTREE_FILE = "kdtree.pkl"
SR          = 22050   # 22050 du dung cho phan tich am nhac, nhe hon 44100
WINDOW_SEC  = 5.0
STEP_SEC    = 2.5
N_WORKERS   = 8


# ─── Trich xuat dac trung ────────────────────────────────────────────────────

def extract_features(audio, sr):
    """
    Trich xuat vector 18 chieu tu audio array.
      [pitch, zcr, energy, centroid, bandwidth, mfcc_1..mfcc_13]
    Sau do chuan hoa ve unit vector (phuc vu Cosine Similarity qua KD-Tree).
    """
    # 1. Pitch — tan so co ban (Hz), nhan dien giai dieu
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )
    pitch = float(np.nanmean(f0[voiced_flag])) if voiced_flag.any() else 0.0

    # 2. Zero-Crossing Rate — toc do doi dau tin hieu (dai dien tan so trung binh)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    # 3. Energy — nang luong trung binh (the hien do to nho)
    energy = float(np.mean(audio ** 2))

    # 4. Spectral Centroid — trong tam pho / do sang (brightness)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    # 5. Spectral Bandwidth — bang thong pho
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))

    # 6. MFCC — 13 he so am sac (mo ta am sac hieu qua nhat)
    mfcc_mean = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1).tolist()

    features = np.array(
        [pitch, zcr, energy, centroid, bandwidth] + mfcc_mean,
        dtype=np.float32
    )

    # Chuan hoa thanh unit vector → tuong duong Cosine Similarity khi dung KD-Tree L2
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features.tolist()


def extract_features_from_file(file_path):
    """Trich xuat dac trung tu file WAV (dung cho query don le)."""
    audio, sr = librosa.load(file_path, sr=SR, mono=True)
    return extract_features(audio, sr)


def get_song_name(file_path):
    """
    Lay ten hien thi tu duong dan file.
    Bo prefix 'IRMAS-TestingData-PartX__PartX__' neu co.
    Vi du: 'IRMAS-TestingData-Part1__Part1__(02) dont kill the whale-4.wav'
           → '(02) dont kill the whale-4'
    """
    import re
    stem = os.path.splitext(os.path.basename(file_path))[0]
    # Bo prefix dang IRMAS-...__...__
    name = re.sub(r'^IRMAS-[^_]+__[^_]+__', '', stem).strip()
    # Neu khong co prefix IRMAS, dung ten goc
    if not name:
        name = stem
    return name


# ─── Sliding Window ──────────────────────────────────────────────────────────

def process_file(file_path):
    """
    Cat file thanh cac cua so truot WINDOW_SEC/STEP_SEC,
    trich xuat dac trung moi cua so.
    Tra ve list (name, file_path, offset_sec, features).
    """
    window_len = int(WINDOW_SEC * SR)
    step_len   = int(STEP_SEC   * SR)
    name  = get_song_name(file_path)

    audio, sr = librosa.load(file_path, sr=SR, mono=True)
    results   = []

    for start in range(0, len(audio) - window_len + 1, step_len):
        window = audio[start : start + window_len]
        feats  = extract_features(window, sr)
        results.append((name, file_path, start / sr, feats))

    return results


# ─── SQLite ──────────────────────────────────────────────────────────────────

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            file_path TEXT NOT NULL,
            offset    REAL NOT NULL,
            features  BLOB NOT NULL
        )
    """)
    conn.commit()


def insert_windows_batch(conn, rows):
    data = [
        (name, fp, offset, pickle.dumps(np.array(feats, dtype=np.float32)))
        for name, fp, offset, feats in rows
    ]
    conn.executemany(
        "INSERT INTO songs (name, file_path, offset, features) VALUES (?,?,?,?)",
        data
    )
    conn.commit()


def load_all_songs(conn):
    rows = conn.execute(
        "SELECT id, name, file_path, offset, features FROM songs"
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "path": r[2], "offset": r[3],
         "features": pickle.loads(r[4])}
        for r in rows
    ]


# ─── KD-Tree ─────────────────────────────────────────────────────────────────

def build_kdtree(songs):
    vectors = np.array([s["features"] for s in songs], dtype=np.float32)
    tree    = KDTree(vectors)
    return tree, vectors


# ─── Build Database ───────────────────────────────────────────────────────────

def build_database(reset=False):
    # --- Reset neu duoc yeu cau ---
    if reset:
        for f in [DB_FILE, KDTREE_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"Da xoa: {f}")

    # --- Kiem tra schema cu (khong co cot offset) → tu dong xoa ---
    if os.path.exists(DB_FILE) and not reset:
        _conn = sqlite3.connect(DB_FILE)
        cols  = [r[1] for r in _conn.execute("PRAGMA table_info(songs)").fetchall()]
        _conn.close()
        if "offset" not in cols:
            print("Phat hien DB cu (khong co cot offset) → xoa va build lai.\n")
            os.remove(DB_FILE)
            if os.path.exists(KDTREE_FILE):
                os.remove(KDTREE_FILE)

    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    existing_paths = set(
        r[0] for r in conn.execute("SELECT DISTINCT file_path FROM songs").fetchall()
    )

    # --- Quet dataset (ho tro ca flat va co thu muc con) ---
    wav_files = sorted(glob.glob(
        os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True
    ) + glob.glob(
        os.path.join(DATASET_DIR, "*.wav")
    ))
    wav_files = sorted(set(
        f for f in wav_files
        if not os.path.basename(f).startswith("_temp")
    ))

    if not wav_files:
        print(f"Khong tim thay file WAV nao trong '{DATASET_DIR}/'")
        conn.close()
        return

    todo    = [f for f in wav_files if f not in existing_paths]
    total   = len(wav_files)
    skipped = total - len(todo)

    print(f"Tim thay {total} file WAV trong '{DATASET_DIR}/'")
    print(f"Bo qua (da co): {skipped} | Can xu ly: {len(todo)}")
    print(f"Window: {WINDOW_SEC}s | Step: {STEP_SEC}s | Workers: {N_WORKERS}\n")

    if todo:
        new_count = 0
        t_start   = time.time()

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(process_file, fp): fp for fp in todo}
            done    = 0

            for future in as_completed(futures):
                fp    = futures[future]
                done += 1
                fname = os.path.basename(fp)
                try:
                    t0   = time.time()
                    rows = future.result()
                    insert_windows_batch(conn, rows)
                    new_count += 1
                    elapsed = time.time() - t0
                    print(f"  [{done:4}/{len(todo)}] {fname[:50]:50s} | {len(rows)} windows | {elapsed:.1f}s")
                except Exception as e:
                    print(f"  [{done:4}/{len(todo)}] {fname}: LOI - {e}")

        print(f"\nHoan thanh trich xuat: {new_count} file moi ({time.time()-t_start:.1f}s)\n")

    # --- Load tat ca va build KD-Tree ---
    songs = load_all_songs(conn)
    conn.close()

    if not songs:
        print("DB trong, khong co gi de build KD-Tree.")
        return

    print(f"Build KD-Tree tu {len(songs)} windows...")
    tree, _ = build_kdtree(songs)
    with open(KDTREE_FILE, "wb") as f:
        pickle.dump({"tree": tree, "songs": songs}, f)

    print(f"\n{'='*55}")
    print(f"Tong bai trong DB  : {len(set(s['name'] for s in songs))}")
    print(f"Tong windows       : {len(songs)}")
    print(f"So chieu vector    : {len(songs[0]['features'])}")
    print(f"KD-Tree            : {KDTREE_FILE}")
    print(f"Database           : {DB_FILE}")


if __name__ == "__main__":
    reset = "--reset" in sys.argv
    build_database(reset=reset)
