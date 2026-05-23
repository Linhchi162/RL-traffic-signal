"""
plot_flow_profile.py — Traffic flow profile from .rou.xml files.

Counts vehicles departing per minute and plots the inflow profile
for Cologne-3 and Cologne-8 side by side.

Output: figures/fig_flow_profile.png
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROU_C3 = "nets/cologne3/cologne3.rou.xml"
ROU_C8 = "nets/cologne8/cologne8.rou.xml"
OUT    = "figures/fig_flow_profile.png"


def load_departs(path: str) -> list[float]:
    tree = ET.parse(path)
    root = tree.getroot()
    departs = []
    for tag in ("vehicle", "trip", "person"):
        for elem in root.findall(f".//{tag}"):
            d = elem.get("depart")
            if d is not None:
                departs.append(float(d))
    return sorted(departs)


def bin_per_minute(departs: list[float], t_start: float, t_end: float):
    """Return (minutes_from_midnight list, counts list) binned per 60 s."""
    n_bins = int((t_end - t_start) / 60) + 1
    counts = [0] * n_bins
    for d in departs:
        idx = int((d - t_start) / 60)
        if 0 <= idx < n_bins:
            counts[idx] += 1
    minutes = [t_start / 60 + i for i in range(n_bins)]
    return minutes, counts


def minutes_to_hhmm(m: float) -> str:
    h = int(m // 60)
    mm = int(m % 60)
    return f"{h:02d}:{mm:02d}"


def main():
    Path("figures").mkdir(exist_ok=True)

    print("Loading Cologne-3 ...", end=" ")
    dep_c3 = load_departs(ROU_C3)
    print(f"{len(dep_c3)} vehicles, {dep_c3[0]/3600:.2f}h – {dep_c3[-1]/3600:.2f}h")

    print("Loading Cologne-8 ...", end=" ")
    dep_c8 = load_departs(ROU_C8)
    print(f"{len(dep_c8)} vehicles, {dep_c8[0]/3600:.2f}h – {dep_c8[-1]/3600:.2f}h")

    # Both maps: 7:00–8:00 only
    t_start = 7 * 3600
    t_end   = 8 * 3600

    mins_c3, cnt_c3 = bin_per_minute(dep_c3, t_start, t_end)
    mins_c8, cnt_c8 = bin_per_minute(dep_c8, t_start, t_end)

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    })

    x_ticks = np.arange(7 * 60, 8 * 60 + 1, 5)   # every 5 min from 7:00 to 8:00

    def make_fig(mins, counts, color, title, out_path):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(mins, counts, where="post", color=color, linewidth=1.8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([minutes_to_hhmm(m) for m in x_ticks], rotation=45, ha="right")
        ax.set_xlim(7 * 60 - 0.5, 8 * 60 + 0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Vehicles per minute", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    make_fig(mins_c3, cnt_c3, "#1976D2",
             "Traffic Inflow Profile — Cologne-3 (07:00–08:00)",
             "figures/fig_flow_c3.png")

    make_fig(mins_c8, cnt_c8, "#E53935",
             "Traffic Inflow Profile — Cologne-8 (07:00–08:00)",
             "figures/fig_flow_c8.png")


if __name__ == "__main__":
    main()
