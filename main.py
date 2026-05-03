import sys
import os


def usage():
    print("""
HE THONG TIM KIEM NHAC THEO NOI DUNG (CBMIR)

Cach dung:
  python main.py build [--reset]           Trich xuat dac trung + build DB
  python main.py search  <file.wav>        Tim 5 bai giong nhat
  python main.py demo_in  <file.wav>       Demo file co trong DB
  python main.py demo_out <file.wav> [ten,ten,...]  Demo file ngoai DB
  python main.py bench    <file.wav>       Benchmark KD-Tree vs Brute-force

Flag:
  --reset   Xoa DB va KD-Tree cu, build lai tu dau (dung khi doi dataset)

Vi du:
  python main.py build
  python main.py build --reset
  python main.py search  dataset/Bai_A.wav
  python main.py demo_in dataset/Bai_A.wav
  python main.py bench   dataset/Bai_A.wav
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(0)

    cmd   = sys.argv[1].lower()
    reset = "--reset" in sys.argv

    # ── Build database ──────────────────────────────────────────────────────
    if cmd == "build":
        from feature_extraction import build_database
        build_database(reset=reset)

    # ── Search ─────────────────────────────────────────────────────────────
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py search <file.wav>")
            sys.exit(1)
        from search import search
        query = sys.argv[2]
        if not os.path.exists(query):
            print(f"Khong tim thay file: {query}")
            sys.exit(1)
        search(query)

    # ── Demo file co trong DB ───────────────────────────────────────────────
    elif cmd == "demo_in":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py demo_in <file.wav>")
            sys.exit(1)
        from evaluate import demo_in_db
        demo_in_db(sys.argv[2])

    # ── Demo file ngoai DB ──────────────────────────────────────────────────
    elif cmd == "demo_out":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py demo_out <file.wav> [ten,ten,...]")
            sys.exit(1)
        from evaluate import demo_out_db
        relevant = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        demo_out_db(sys.argv[2], relevant_names=relevant)

    # ── Benchmark ───────────────────────────────────────────────────────────
    elif cmd == "bench":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py bench <file.wav>")
            sys.exit(1)
        from evaluate import demo_benchmark
        demo_benchmark(sys.argv[2])

    else:
        print(f"Lenh khong hop le: '{cmd}'")
        usage()
