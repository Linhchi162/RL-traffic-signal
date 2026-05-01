"""
make_random_routes.py — Tao route file synthetic (random OD) cho curriculum training.

Su dung SUMO randomTrips.py de sinh luong xe ngau nhien tren mang,
voi so luong xe tuong duong cac giai doan curriculum cua route file that.

Output: <net_dir>/<stem>_rand_25pct.rou.xml, _50pct, _75pct

Su dung:
    python make_random_routes.py --net nets/cologne3/cologne3.net.xml \
        --rou nets/cologne3/cologne3.rou.xml
    python make_random_routes.py --net nets/cologne8/cologne8.net.xml \
        --rou nets/cologne8/cologne8.rou.xml
"""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    sys.exit("[make_random_routes] SUMO_HOME chua duoc khai bao")

RANDOM_TRIPS = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"
FRACTIONS    = [0.25, 0.50, 0.75, 1.00]
BEGIN        = 25200.0
END          = 28800.0


def count_vehicles_in_window(rou_path: Path, begin: float, end: float) -> int:
    """Dem so luong xe bat dau trong [begin, end] trong route file that."""
    root = ET.parse(str(rou_path)).getroot()
    n = 0
    for tag in ("vehicle", "trip"):
        n += sum(
            1 for c in root
            if c.tag == tag and begin <= float(c.get("depart", 0)) <= end
        )
    # flow elements: uoc tinh tu vTimeStep va so vehicle
    for flow in root.iter("flow"):
        fb = float(flow.get("begin", 0))
        fe = float(flow.get("end", 0))
        if fb <= end and fe >= begin:
            dur    = min(fe, end) - max(fb, begin)
            period = float(flow.get("period", flow.get("vehsPerHour", 360)))
            if flow.get("vehsPerHour"):
                period = 3600.0 / float(flow.get("vehsPerHour"))
            n += max(1, int(dur / period))
    return n


def make_rand_route(net_file: Path, out_path: Path,
                    n_vehs: int, begin: float, end: float, seed: int):
    """Goi randomTrips.py de tao route file synthetic voi n_vehs xe."""
    period = (end - begin) / n_vehs

    cmd = [
        sys.executable, str(RANDOM_TRIPS),
        "-n", str(net_file),
        "-r", str(out_path),        # output routed .rou.xml (goi duarouter)
        "--begin",  str(int(begin)),
        "--end",    str(int(end)),
        "-p",       f"{period:.4f}",
        "--seed",   str(seed),
        "--random-depart",
        "--validate",               # bo qua chuyen tuyen that bai
    ]

    print(f"  Goi randomTrips: {n_vehs} xe, period={period:.2f}s ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Thu lai khong co --validate (SUMO cu hon co the khong ho tro)
        cmd_noval = [c for c in cmd if c != "--validate"]
        result = subprocess.run(cmd_noval, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] randomTrips loi: {result.stderr[:300]}", flush=True)
        return False
    print(f"  -> {out_path.name}", flush=True)
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--net",       required=True, help="Network .net.xml")
    p.add_argument("--rou",       required=True, help="Real route file (de dem xe)")
    p.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    p.add_argument("--begin",     type=float, default=BEGIN)
    p.add_argument("--end",       type=float, default=END)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    net_file = Path(args.net)
    rou_file = Path(args.rou)
    stem     = rou_file.stem.replace(".rou", "")

    n_real = count_vehicles_in_window(rou_file, args.begin, args.end)
    if n_real == 0:
        sys.exit(f"[ERR] Khong dem duoc xe trong [{args.begin},{args.end}] tu {rou_file}")

    print("=" * 60)
    print(f"  make_random_routes.py")
    print(f"  Net  : {net_file.name}")
    print(f"  Route: {rou_file.name}  ({n_real} xe trong cua so)")
    print(f"  Seed : {args.seed}")
    print("=" * 60)

    ok = 0
    for frac in sorted(args.fractions):
        pct    = int(round(frac * 100))
        n_vehs = max(1, round(n_real * frac))
        out    = rou_file.parent / f"{stem}_rand_{pct}pct.rou.xml"

        if out.exists():
            print(f"  {pct}%: {out.name} da co, bo qua.")
            ok += 1
            continue

        print(f"\n  [{pct}%] Target: {n_vehs} xe")
        if make_rand_route(net_file, out, n_vehs, args.begin, args.end,
                           args.seed + pct):
            ok += 1
        else:
            print(f"  [WARN] {pct}% that bai — kiem tra duarouter da cai chua.")

    print(f"\nHoan tat: {ok}/{len(args.fractions)} file duoc tao.")


if __name__ == "__main__":
    main()
