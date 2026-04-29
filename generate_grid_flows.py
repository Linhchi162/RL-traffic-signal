"""
generate_grid_flows.py
Sinh SUMO flow file cho 2x2 grid.

Usage:
    python generate_grid_flows.py --seed 0 --out generated_flows/grid_s0.xml
"""

import argparse
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Routes: edge sequences for all main through-movements
# ---------------------------------------------------------------------------

ROUTES = {
    # N-S / S-N through (single column)
    "route_n00_s10": "n00_t00 t00_t10 t10_s10",
    "route_n01_s11": "n01_t01 t01_t11 t11_s11",
    "route_s10_n00": "s10_t10 t10_t00 t00_n00",
    "route_s11_n01": "s11_t11 t11_t01 t01_n01",
    # E-W / W-E through (single row)
    "route_w00_e01": "w00_t00 t00_t01 t01_e01",
    "route_w10_e11": "w10_t10 t10_t11 t11_e11",
    "route_e01_w00": "e01_t01 t01_t00 t00_w00",
    "route_e11_w10": "e11_t11 t11_t10 t10_w10",
    # Diagonal N->E (left turn at junction)
    "route_n00_e01": "n00_t00 t00_t01 t01_e01",
    "route_n01_e11": "n01_t01 t01_t11 t11_e11",
    "route_n00_e11": "n00_t00 t00_t01 t01_t11 t11_e11",   # cross-grid
    # Diagonal N->W (right turn at junction)
    "route_n00_w00": "n00_t00 t00_w00",
    "route_n01_w00": "n01_t01 t01_t00 t00_w00",
    # S->E / S->W
    "route_s10_e11": "s10_t10 t10_t11 t11_e11",
    "route_s11_w10": "s11_t11 t11_t10 t10_w10",
    "route_s10_n01": "s10_t10 t10_t00 t00_t01 t01_n01",   # cross-grid
    # W->S / W->N
    "route_w00_s10": "w00_t00 t00_t10 t10_s10",
    "route_w10_n00": "w10_t10 t10_t00 t00_n00",
    # E->S / E->N
    "route_e01_s11": "e01_t01 t01_t11 t11_s11",
    "route_e11_n01": "e11_t11 t11_t01 t01_n01",
}

# Nhom ong dan theo truc
NS_ROUTES = [r for r in ROUTES if r.startswith(("route_n","route_s"))]
WE_ROUTES = [r for r in ROUTES if r.startswith(("route_w","route_e"))]

VTYPES_XML = """\
<vTypeDistribution id="het_ns">
    <vType id="Car_ns" accel="3.28" decel="7.5" length="4.0" width="1.6"
    height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1"
    vClass="passenger" guiShape="passenger" probability="0.10" color="1,1,0"/>
</vTypeDistribution>
<vTypeDistribution id="het_we">
    <vType id="Car_we" accel="3.28" decel="7.5" length="4.0" width="1.6"
    height="1.5" maxSpeed="33.33" minGap="0.38" sigma="0.1" tau="1.1"
    vClass="passenger" guiShape="passenger" probability="0.10" color="0,1,1"/>
</vTypeDistribution>
"""

SCENARIO_DEMAND = {
    "we_dominant":   {"WE": (1900, 2200), "NS": (800,  1100)},
    "ns_dominant":   {"WE": (800,  1100), "NS": (1900, 2200)},
    "high_demand":   {"WE": (1600, 1800), "NS": (1600, 1800)},
    "balanced_high": {"WE": (1200, 1400), "NS": (1200, 1400)},
    "balanced_low":  {"WE": (600,   800), "NS": (600,   800)},
    "low_demand":    {"WE": (300,   450), "NS": (300,   450)},
}


def get_scenario_sequence(rng, n_blocks):
    pool = list(SCENARIO_DEMAND.keys())
    base = list(pool); rng.shuffle(base)
    extra = [rng.choice(pool) for _ in range(max(0, n_blocks - len(pool)))]
    seq = (base + extra)[:n_blocks]; rng.shuffle(seq)
    return seq


def generate_grid_flows(
    seed=42,
    interval_sec=145,
    total_duration_sec=100_000,
    block_intervals=30,
    output_path="generated_flows/grid_s0.xml",
    scale=1.0,
):
    rng = random.Random(seed)
    n_intervals = total_duration_sec // interval_sec
    n_blocks = math.ceil(n_intervals / block_intervals)
    scenarios = get_scenario_sequence(rng, n_blocks)

    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        VTYPES_XML,
    ]
    for rid, edges in ROUTES.items():
        lines.append(f'    <route id="{rid}" edges="{edges}" />')
    lines.append("")

    flow_counter = 0
    for iv in range(n_intervals):
        t_begin = iv * interval_sec
        t_end   = t_begin + interval_sec
        block   = iv // block_intervals
        scen    = scenarios[min(block, len(scenarios) - 1)]
        demand  = SCENARIO_DEMAND[scen]

        phase = (iv % block_intervals) / block_intervals
        tf = 0.85 + 0.30 * math.sin(2 * math.pi * phase)

        ns_lo, ns_hi = demand["NS"]
        we_lo, we_hi = demand["WE"]
        ns_total = rng.uniform(ns_lo, ns_hi) * tf * scale
        we_total = rng.uniform(we_lo, we_hi) * tf * scale

        for rid in ROUTES:
            is_ns = rid in NS_ROUTES
            vph = rng.uniform(0.3, 0.7) * (ns_total if is_ns else we_total)
            vph = max(0.0, vph * rng.uniform(0.85, 1.15))
            if vph < 1.0:
                continue
            vtype = "het_ns" if is_ns else "het_we"
            slot = iv + 1
            lines.append(
                f'    <flow id="flow_{rid[6:]}_{slot}" type="{vtype}" '
                f'route="{rid}" begin="{t_begin}" end="{t_end}" '
                f'vehsPerHour="{vph:.4f}" />'
            )
            flow_counter += 1

    lines.append("</routes>")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out.name} — {n_intervals} intervals {flow_counter} flows seed={seed}")
    return str(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out",      default="generated_flows/grid_s0.xml")
    p.add_argument("--duration", type=int, default=100_000)
    p.add_argument("--interval", type=int, default=145)
    p.add_argument("--block",    type=int, default=30)
    p.add_argument("--scale",    type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_grid_flows(
        seed=args.seed,
        interval_sec=args.interval,
        total_duration_sec=args.duration,
        block_intervals=args.block,
        output_path=args.out,
        scale=args.scale,
    )
