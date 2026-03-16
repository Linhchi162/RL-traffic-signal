"""
generate_test_flows.py — Sinh route files kiem tra voi 3 muc mat do luu luong

Su dung:
    python sumo_nets/generate_test_flows.py

Output:
    sumo_nets/grid2x2_test_high.rou.xml   -- High flow  (period=5,  ~720 xe/h/pair)
    sumo_nets/grid2x2_test_medium.rou.xml -- Medium flow (period=9,  ~400 xe/h/pair)
    sumo_nets/grid2x2_test_low.rou.xml    -- Low flow   (period=18, ~200 xe/h/pair)

Ghi chu:
    Seed khac hoan toan voi tap train (seed=42) de dam bao tinh generalization.
    High/Medium/Low tuong ung voi 3 kich ban kiem tra trong bai bao RLTSCQ.
"""

import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SUMO_HOME = os.environ.get("SUMO_HOME", "")
if not SUMO_HOME:
    sys.exit("Loi: Chua khai bao SUMO_HOME")

HERE = Path(__file__).parent
NET_FILE = HERE / "grid2x2.net.xml"

# (ten_file, period, seed, label)
FLOW_CONFIGS = [
    ("grid2x2_test_high.rou.xml",   5.0,  200, "high"),    # ~720 xe/h/pair
    ("grid2x2_test_medium.rou.xml", 9.0,  123, "medium"),  # ~400 xe/h/pair (bat buoc phai co file nay)
    ("grid2x2_test_low.rou.xml",   18.0,  456, "low"),     # ~200 xe/h/pair
]

TEST_DURATION = 5_200   # 5000s test + 200s warmup


def find_random_trips():
    candidates = [
        Path(SUMO_HOME) / "tools" / "randomTrips.py",
        Path(SUMO_HOME) / "tools" / "trip" / "randomTrips.py",
        Path(SUMO_HOME) / "share" / "sumo" / "tools" / "randomTrips.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"Khong tim thay randomTrips.py trong {SUMO_HOME}")


def generate_flow(route_file: Path, period: float, seed: int, label: str):
    if not NET_FILE.exists():
        sys.exit(
            f"Chua co file mang {NET_FILE.name}.\n"
            "Hay chay truoc: python sumo_nets/generate_net.py"
        )

    random_trips = find_random_trips()
    cmd = [
        sys.executable, str(random_trips),
        "--net-file", str(NET_FILE),
        "--route-file", str(route_file),
        "--begin", "0",
        "--end", str(TEST_DURATION),
        "--period", str(period),
        "--seed", str(seed),
        "--min-distance", "100",
        "--allow-fringe",
        "--validate",
    ]
    veh_per_hour = int(3600 / period)
    print(f"[{label.upper():6}] Tao {route_file.name}  "
          f"(period={period}s, ~{veh_per_hour} xe/h/pair)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:300]}")
        print(f"  Canh bao: randomTrips ket thuc voi code {result.returncode}")
    else:
        size_kb = round(route_file.stat().st_size / 1024, 1)
        print(f"  OK — {route_file.name} ({size_kb} KB)")


def main():
    print("=" * 55)
    print("  Sinh route files kiem tra — 3 muc luu luong")
    print(f"  Mang luoi : {NET_FILE.name}")
    print(f"  Thoi gian : {TEST_DURATION}s moi file")
    print("=" * 55)

    for fname, period, seed, label in FLOW_CONFIGS:
        out_path = HERE / fname
        if out_path.exists():
            size_kb = round(out_path.stat().st_size / 1024, 1)
            print(f"[{label.upper():6}] Bo qua — {fname} da ton tai ({size_kb} KB)")
            continue
        generate_flow(out_path, period, seed, label)

    print()
    print("=" * 55)
    print("  Hoan tat! Files trong sumo_nets/")
    for fname, period, _, label in FLOW_CONFIGS:
        p = HERE / fname
        exists = "OK" if p.exists() else "MISSING"
        print(f"  [{label.upper():6}] {fname}  [{exists}]")
    print("=" * 55)


if __name__ == "__main__":
    main()
