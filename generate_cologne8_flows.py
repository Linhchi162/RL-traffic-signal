"""
generate_cologne8_flows.py
Sinh route file cho Cologne8 voi luong xe trung binh cua 6 kich ban train.

Trung binh 6 kich ban (WE + NS, ca 2 truc):
  we_dominant:   (2050+950)/2  = 1500
  ns_dominant:   (950+2050)/2  = 1500
  high_demand:   1700
  balanced_high: 1300
  balanced_low:  700
  low_demand:    375
  => Trung binh: ~1179 veh/hr

Dung SUMO randomTrips.py (co --validate de duarouter tinh route hop le).

Usage:
    python generate_cologne8_flows.py
    python generate_cologne8_flows.py --vph 1179 --seed 42
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
NET  = HERE / "nets" / "cologne8" / "cologne8.net.xml"
OUT  = HERE / "nets" / "cologne8" / "cologne8_synthetic.rou.xml"

AVG_VPH = 1179   # trung binh 6 kich ban train
BEGIN   = 25200  # 7:00 AM
END     = 28800  # 8:00 AM


def find_random_trips() -> str:
    candidates = [
        os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py"),
        "/usr/share/sumo/tools/randomTrips.py",
        "/usr/local/share/sumo/tools/randomTrips.py",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net",   default=str(NET),  help="Duong dan cologne8.net.xml")
    p.add_argument("--out",   default=str(OUT),  help="File output .rou.xml")
    p.add_argument("--vph",   type=float, default=AVG_VPH,
                   help="Vehicles per hour (default: ~1179 = avg 6 train scenarios)")
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--begin", type=int,   default=BEGIN)
    p.add_argument("--end",   type=int,   default=END)
    args = p.parse_args()

    if not os.path.isfile(args.net):
        sys.exit(f"[ERR] Khong tim thay: {args.net}\n"
                 f"      Chay: bash setup_vast.sh  de download Cologne8 files")

    random_trips = find_random_trips()
    if not random_trips:
        sys.exit("[ERR] Khong tim thay randomTrips.py — kiem tra SUMO_HOME")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    period = 3600.0 / args.vph   # giay giua 2 xe lien tiep
    n_total = int(args.vph * (args.end - args.begin) / 3600)

    print(f"=== generate_cologne8_flows ===")
    print(f"Demand : {args.vph:.0f} veh/hr  (~{n_total} xe trong {args.end-args.begin}s)")
    print(f"Period : {period:.3f}s giua 2 xe")
    print(f"Window : {args.begin}s - {args.end}s")
    print(f"Output : {args.out}")
    print()

    cmd = [
        sys.executable, random_trips,
        "-n", args.net,
        "-o", args.out,
        "-b", str(args.begin),
        "-e", str(args.end),
        "-p", f"{period:.4f}",
        "--seed", str(args.seed),
        "--vehicle-class", "passenger",
        "--validate",          # chay qua duarouter de dam bao route hop le
        "--remove-loops",
        "--min-distance", "100",
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[ERR] randomTrips.py that bai (exit {result.returncode})")

    print(f"\n[OK] {args.out}")
    print(f"     Chay eval: python evaluate_cologne8.py --route_file {args.out}")


if __name__ == "__main__":
    main()
