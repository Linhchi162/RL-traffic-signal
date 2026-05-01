"""
make_scaled_routes.py — Tao route file voi demand giam dan cho curriculum learning.

Lay ngau nhien mot phan xe trong cua so thoi gian chi dinh (mac dinh 7-8 AM),
giu nguyen tat ca xe ngoai cua so. Output: <name>_25pct.rou.xml, v.v.

Su dung:
    python make_scaled_routes.py --rou nets/cologne3/cologne3.rou.xml
    python make_scaled_routes.py --rou nets/cologne3/cologne3.rou.xml \
        --fractions 0.25 0.5 0.75 --begin 25200 --end 28800 --seed 42
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def scale_route(input_path: Path, output_path: Path,
                fraction: float, begin: float, end: float, seed: int):
    ET.register_namespace("", "")
    tree = ET.parse(str(input_path))
    root = tree.getroot()

    all_vehs = [c for c in root if c.tag == "vehicle"]

    # Phan loai xe trong / ngoai cua so thoi gian
    in_window  = [v for v in all_vehs
                  if begin <= float(v.get("depart", 0)) <= end]
    out_window = [v for v in all_vehs
                  if not (begin <= float(v.get("depart", 0)) <= end)]

    # Sample xe trong cua so
    rng    = random.Random(seed)
    n_keep = max(1, round(len(in_window) * fraction))
    keep   = set(id(v) for v in rng.sample(in_window, n_keep))

    # Xoa xe khong duoc chon
    for v in in_window:
        if id(v) not in keep:
            root.remove(v)

    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
    return n_keep, len(in_window)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rou",       required=True, help="Input .rou.xml")
    p.add_argument("--fractions", nargs="+", type=float,
                   default=[0.25, 0.5, 0.75],
                   help="Demand fractions (1.0 = original, da co san)")
    p.add_argument("--begin",     type=float, default=25200.0)
    p.add_argument("--end",       type=float, default=28800.0)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main():
    args     = parse_args()
    src      = Path(args.rou)
    stem     = src.stem.replace(".rou", "").replace(".net", "")

    print(f"Input : {src}")
    print(f"Window: {args.begin}–{args.end}s  seed={args.seed}")
    print()

    # Count vehicles in window for info
    tree = ET.parse(str(src))
    root = tree.getroot()
    n_full = sum(1 for c in root if c.tag == "vehicle"
                 and args.begin <= float(c.get("depart", 0)) <= args.end)
    print(f"Vehicles in window: {n_full}")
    print()

    for frac in sorted(args.fractions):
        pct     = int(round(frac * 100))
        out     = src.parent / f"{stem}_{pct}pct.rou.xml"
        n, total = scale_route(src, out, frac, args.begin, args.end, args.seed)
        print(f"  {pct:3d}%  →  {n:4d}/{total} vehicles  →  {out.name}")

    print("\nHoan tat.")


if __name__ == "__main__":
    main()
