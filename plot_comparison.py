"""Generate comparison plots for STMARL vs Fixed-time baseline."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from evaluate_metrics import parse_tripinfo


def compare_att(
    stmarl_tripinfo: str,
    baseline_tripinfo: str,
    output_file: str = "comparison_results.png",
) -> bool:
    """
    Compare Average Travel Time (ATT) between STMARL and Fixed-time baseline.
    
    Args:
        stmarl_tripinfo: Path to tripinfo.xml from STMARL AI run
        baseline_tripinfo: Path to tripinfo.xml from fixed-time baseline run
        output_file: Output filename for comparison plot
    """
    stmarl_tripinfo = Path(stmarl_tripinfo)
    baseline_tripinfo = Path(baseline_tripinfo)
    
    # Parse both results
    print("📊 Parsing STMARL results...")
    stmarl_metrics = parse_tripinfo(stmarl_tripinfo)
    if not stmarl_metrics:
        print("❌ Failed to parse STMARL tripinfo")
        return False
    
    print("📊 Parsing Baseline (Fixed-time) results...")
    baseline_metrics = parse_tripinfo(baseline_tripinfo)
    if not baseline_metrics:
        print("❌ Failed to parse Baseline tripinfo")
        return False
    
    # Create comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ["STMARL\n(AI-based)", "Fixed-time\n(Baseline)"]
    att_values = [stmarl_metrics.avg_travel_time, baseline_metrics.avg_travel_time]
    colors = ["#2ecc71", "#e74c3c"]  # Green for AI, Red for baseline
    
    bars = ax.bar(methods, att_values, color=colors, alpha=0.8, edgecolor="black", linewidth=2)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, att_values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    
    # Add improvement percentage
    improvement = ((baseline_metrics.avg_travel_time - stmarl_metrics.avg_travel_time) 
                   / baseline_metrics.avg_travel_time * 100)
    ax.text(
        0.5,
        max(att_values) * 0.9,
        f"✨ Improvement: {improvement:.1f}%",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )
    
    ax.set_ylabel("Average Travel Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("STMARL vs Fixed-time Traffic Signal Control", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(att_values) * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Comparison plot saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("📈 COMPARISON SUMMARY")
    print("="*50)
    print(f"\n🤖 STMARL (AI):")
    print(f"   - ATT: {stmarl_metrics.avg_travel_time:.2f}s")
    print(f"   - Total trips: {stmarl_metrics.total_trips}")
    print(f"\n🔄 Fixed-time (Baseline):")
    print(f"   - ATT: {baseline_metrics.avg_travel_time:.2f}s")
    print(f"   - Total trips: {baseline_metrics.total_trips}")
    print(f"\n✨ Improvement: {improvement:.1f}%")
    print(f"   Time saved per trip: {baseline_metrics.avg_travel_time - stmarl_metrics.avg_travel_time:.2f}s")
    print("="*50 + "\n")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <stmarl_tripinfo.xml> <baseline_tripinfo.xml> [output.png]")
        print("\nExample:")
        print("  python plot_comparison.py scenarios/grid_3x3/tripinfos_stmarl.xml scenarios/grid_3x3/tripinfos_baseline.xml")
        sys.exit(1)
    
    stmarl_file = sys.argv[1]
    baseline_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "comparison_results.png"
    
    compare_att(stmarl_file, baseline_file, output_file)
