"""
generate_train_flows.py
Sinh SUMO flow file moi cho training voi luu luong ngau nhien hoa.

Muc tieu:
  - Cover ca NS-dominant, WE-dominant va balanced traffic
  - Time-varying demand (peak / off-peak) de agent hoc policy tong quat

Nguon so lieu:
  - Luu luong theo TCCS 24:2018 (quy doi PCU/h)
  - Capacity 1 lan ~1800-1900 PCU/h; thiet ke cho nut giao 2-3 lan/huong
  - Ti le re trai/phai theo khao sat giao thong thuc te Viet Nam

Usage:
    python generate_train_flows.py --seed 0   --out generated_flows/train_s0.xml
    python generate_train_flows.py --seed 1   --out generated_flows/train_s1.xml
    python generate_train_flows.py --seed 2   --out generated_flows/train_s2.xml
    python generate_train_flows.py --balanced --out generated_flows/train_balanced.xml
"""

import argparse
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Cau truc giao lo: 4 huong, moi huong 3 luong xe (straight / right / left)
# ---------------------------------------------------------------------------
# Flow ID prefix  | vType                  | route_id
MOVEMENTS = [
    # North approach
    ("NS", "heterogeneous_traffic_ns", "route_ns"),   # straight
    ("NR", "heterogeneous_traffic_ns", "route_nw"),   # right
    ("NL", "heterogeneous_traffic_ns", "route_ne"),   # left
    # South approach
    ("SS", "heterogeneous_traffic_sw", "route_sn"),
    ("SR", "heterogeneous_traffic_sw", "route_se"),
    ("SL", "heterogeneous_traffic_sw", "route_sw"),
    # West approach
    ("WS", "heterogeneous_traffic_nw", "route_we"),
    ("WR", "heterogeneous_traffic_nw", "route_wn"),
    ("WL", "heterogeneous_traffic_nw", "route_ws"),
    # East approach
    ("ES", "heterogeneous_traffic_ne", "route_ew"),
    ("ER", "heterogeneous_traffic_ne", "route_es"),
    ("EL", "heterogeneous_traffic_ne", "route_en"),
]

# Nhom huong: NS = {N, S}, WE = {W, E}
NS_PREFIXES = {"NS", "NR", "NL", "SS", "SR", "SL"}
WE_PREFIXES = {"WS", "WR", "WL", "ES", "ER", "EL"}

VTYPES_XML = """\
<vTypeDistribution id="heterogeneous_traffic_ns">
    <vType id="Car_NS" accel="3.28" decel="7.5" length="4.0" width="1.6" \
height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1" \
vClass="passenger" guiShape="passenger" probability="0.10" color="1,1,0" />
</vTypeDistribution>

<vTypeDistribution id="heterogeneous_traffic_nw">
    <vType id="Car_NW" accel="3.28" decel="7.5" length="4.0" width="1.6" \
height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1" \
vClass="passenger" guiShape="passenger" probability="0.08" color="1,1,0" />
</vTypeDistribution>

<vTypeDistribution id="heterogeneous_traffic_ne">
    <vType id="Car_NE" accel="3.28" decel="7.5" length="4.0" width="1.6" \
height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1" \
vClass="passenger" guiShape="passenger" probability="0.21" color="1,1,0" />
</vTypeDistribution>

<vTypeDistribution id="heterogeneous_traffic_sw">
    <vType id="Car_SW" accel="3.28" decel="7.5" length="4.0" width="1.6" \
height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1" \
vClass="passenger" guiShape="passenger" probability="0.08" color="1,1,0" />
</vTypeDistribution>
"""

ROUTES_XML = """\
    <route id="route_ns" edges="n_t t_s" />
    <route id="route_nw" edges="n_t t_w" />
    <route id="route_ne" edges="n_t t_e" />
    <route id="route_we" edges="w_t t_e" />
    <route id="route_wn" edges="w_t t_n" />
    <route id="route_ws" edges="w_t t_s" />
    <route id="route_ew" edges="e_t t_w" />
    <route id="route_en" edges="e_t t_n" />
    <route id="route_es" edges="e_t t_s" />
    <route id="route_sn" edges="s_t t_n" />
    <route id="route_se" edges="s_t t_e" />
    <route id="route_sw" edges="s_t t_w" />
"""


# ---------------------------------------------------------------------------
# Cac thong so luu luong (PCU/h) theo TCCS 24:2018
# ---------------------------------------------------------------------------

# Moi entry la (lo, hi) PCU/h cho TUNG approach (N, S, W, E rieng biet).
# Capacity 1 lan ~1800-1900 PCU/h; nut giao 2-3 lan => tong ~3600-5700 PCU/h.
# Cac muc demand duoi day dam bao khong gay gridlock ao.
SCENARIO_DEMAND = {
    # Gio cao diem sang/chieu bat doi xung (mo phong nut giao theo TCCS 24:2018)
    "we_dominant":   {"WE": (1900, 2200), "NS": (800,  1100)},
    "ns_dominant":   {"WE": (800,  1100), "NS": (1900, 2200)},

    # Gio cao diem le tet - ket xe toan dien
    "high_demand":   {"WE": (1600, 1800), "NS": (1600, 1800)},

    # Gio hanh chinh binh thuong
    "balanced_high": {"WE": (1200, 1400), "NS": (1200, 1400)},
    "balanced_low":  {"WE": (600,   800), "NS": (600,   800)},

    # Buoi toi / dem - vang ve
    "low_demand":    {"WE": (300,   450), "NS": (300,   450)},
}

# Ti le re theo khao sat giao thong thuc te Viet Nam (TCCS 24:2018)
# Sample doc lap theo tung loai, sau do normalize de tong = 1.
TURN_RATIOS = {
    "straight": (0.80, 0.90),   # Di thang: 80-90% (khop khao sat thuc te)
    "right":    (0.05, 0.12),   # Re phai:  5-12%
    "left":     (0.05, 0.08),   # Re trai:  5-8%  (dong gay ket xe chinh)
}


def sample_approach_demands(rng: random.Random, total_approach: float):
    """Tra ve (straight, right, left) PCU/h tu tong luu luong 1 approach."""
    s = rng.uniform(*TURN_RATIOS["straight"])
    r = rng.uniform(*TURN_RATIOS["right"])
    l = rng.uniform(*TURN_RATIOS["left"])
    total = s + r + l
    s_ratio, r_ratio, l_ratio = s / total, r / total, l / total

    noise = lambda x: max(0.0, x * rng.uniform(0.92, 1.08))
    return (noise(total_approach * s_ratio),
            noise(total_approach * r_ratio),
            noise(total_approach * l_ratio))


def get_scenario_sequence(rng: random.Random, n_blocks: int,
                          force_balanced: bool = False):
    """
    Sinh danh sach kich ban cho tung 'block' thoi gian (~60-120 phut moi block).
    Muc dich: dam bao training cover ca NS-dominant, WE-dominant va balanced.
    """
    if force_balanced:
        scenarios = ["balanced_high", "balanced_low"] * (n_blocks // 2 + 1)
        return scenarios[:n_blocks]

    pool = list(SCENARIO_DEMAND.keys())
    # Dam bao moi loai xuat hien it nhat 1 lan
    base = list(pool)
    rng.shuffle(base)
    extra = [rng.choice(pool) for _ in range(max(0, n_blocks - len(pool)))]
    sequence = (base + extra)[:n_blocks]
    rng.shuffle(sequence)
    return sequence


def generate_flows(
    seed: int = 42,
    interval_sec: int = 145,          # do dai moi time slot (giay)
    total_duration_sec: int = 100_000, # tong thoi gian simulation
    block_intervals: int = 30,         # so interval moi block (~72 phut)
    force_balanced: bool = False,
    output_path: str = "generated_train_flows.xml",
    scale: float = 1.0,               # nhan vao tat ca demand (0.4 ~ train_flows.xml)
):
    rng = random.Random(seed)

    n_intervals = total_duration_sec // interval_sec
    n_blocks = math.ceil(n_intervals / block_intervals)
    scenarios = get_scenario_sequence(rng, n_blocks, force_balanced)

    lines = []
    lines.append("<?xml version='1.0' encoding='utf-8'?>")
    lines.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    lines.append("")
    lines.append(VTYPES_XML)
    lines.append(ROUTES_XML)

    flow_counter = 0
    for interval_idx in range(n_intervals):
        t_begin = interval_idx * interval_sec
        t_end   = t_begin + interval_sec

        block_idx = interval_idx // block_intervals
        scenario  = scenarios[min(block_idx, len(scenarios) - 1)]
        demand    = SCENARIO_DEMAND[scenario]

        # Lay tong luu luong cho moi approach trong block nay
        # Them bien dong theo thoi gian trong block (sine wave nho)
        phase = (interval_idx % block_intervals) / block_intervals
        time_factor = 0.85 + 0.30 * math.sin(2 * math.pi * phase)

        ns_lo, ns_hi = demand["NS"]
        we_lo, we_hi = demand["WE"]
        total_N = rng.uniform(ns_lo, ns_hi) * time_factor * scale
        total_S = rng.uniform(ns_lo, ns_hi) * time_factor * scale
        total_W = rng.uniform(we_lo, we_hi) * time_factor * scale
        total_E = rng.uniform(we_lo, we_hi) * time_factor * scale

        approach_totals = {
            "N": total_N, "S": total_S,
            "W": total_W, "E": total_E,
        }

        # Tinh tung luong xe
        # movement prefix: NS/NR/NL=N, SS/SR/SL=S, WS/WR/WL=W, ES/ER/EL=E
        demands_per_move = {}
        sampled = {}
        for approach, total in approach_totals.items():
            s, r, l = sample_approach_demands(rng, total)
            sampled[approach] = (s, r, l)   # (straight, right, left)

        # Map prefix -> demand value
        move_demand = {}
        for prefix, vtype, route in MOVEMENTS:
            approach = prefix[0]   # N, S, W, E
            idx = {"S": 0, "R": 1, "L": 2}[prefix[1]]
            move_demand[prefix] = sampled[approach][idx]

        # Ghi flow entries
        slot = interval_idx + 1
        for prefix, vtype, route in MOVEMENTS:
            vph = move_demand[prefix]
            if vph < 1.0:
                continue
            flow_id = f"flow_{prefix}_{slot}"
            lines.append(
                f'    <flow id="{flow_id}" type="{vtype}" route="{route}" '
                f'begin="{t_begin}" end="{t_end}" vehsPerHour="{vph:.4f}" />'
            )
            flow_counter += 1

    lines.append("</routes>")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {out.name}  —  {n_intervals} intervals  {flow_counter} flows  "
          f"seed={seed}  balanced={force_balanced}")
    return str(out)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out",      default="generated_flows/train_mixed.xml")
    p.add_argument("--duration", type=int, default=100_000,
                   help="Tong thoi gian simulation (giay)")
    p.add_argument("--interval", type=int, default=145,
                   help="Do dai moi time slot (giay)")
    p.add_argument("--block",    type=int, default=30,
                   help="So time slots moi traffic block (~30*145s = 73 phut)")
    p.add_argument("--balanced", action="store_true",
                   help="Chi sinh balanced traffic (test khong co distribution shift)")
    p.add_argument("--scale", type=float, default=1.0,
                   help="He so nhan vao tat ca demand (0.4 ~ mat do train_flows.xml)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_flows(
        seed=args.seed,
        interval_sec=args.interval,
        total_duration_sec=args.duration,
        block_intervals=args.block,
        force_balanced=args.balanced,
        output_path=args.out,
        scale=args.scale,
    )
