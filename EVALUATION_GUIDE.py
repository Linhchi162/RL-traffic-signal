"""
STMARL EVALUATION & REPORTING GUIDE
====================================

3-step workflow to collect data and generate thesis conclusions.
"""

# ============================================================================
# STEP 1: Measure STMARL Performance (AI-based)
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ STEP 1: RUN STMARL TEST & COLLECT METRICS                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Prerequisite:
- Trained model weights already saved as: stmarl_weights.pth
- SUMO config already has: <tripinfo-output value="tripinfos.xml"/>

Command to run:
───────────────────────────────────────────────────────────────────────────

    .\.venv\Scripts\Activate.ps1
    python train_gnn_lstm_dqn.py \\
        --cfg scenarios/grid_3x3_lanes3_dynamic_dense/run.sumo.cfg \\
        --episodes 5 \\
        --max-decisions 400 \\
        --seq-len 5 \\
        --eps-start 0 \\
        --eps-end 0 \\
        --gui

Parameters:
  - eps-start=0, eps-end=0  : Pure exploitation (no exploration)
  - episodes=5              : Run 5 independent tests (or more for stability)
  - gui                     : Optional, watch the Green Waves in action!

Output file created:
  📁 scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos.xml

   
Calculate Average Travel Time:
───────────────────────────────────────────────────────────────────────────

    python evaluate_metrics.py scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos.xml

Example output:
    📊 Trip Metrics:
      Total trips: 1245
      Departed: 1245
      Arrived: 1180
      Total duration sum: 48520.45s
      ⏱️  Average Travel Time: 39.05s

✅ RECORD THIS NUMBER: 39.05s (STMARL ATT)


╔════════════════════════════════════════════════════════════════════════════╗
║ STEP 2: RUN BASELINE TEST (Fixed-time Control)                            ║
╚════════════════════════════════════════════════════════════════════════════╝

This is the "dumb" system to compare against. SUMO auto-cycles signals.

Command to run:
───────────────────────────────────────────────────────────────────────────

    python run_baseline_fixed_time.py --gui

What happens:
  - SUMO starts with default fixed-time control
  - Vehicles follow same routes (from grid_3x3*.rou.xml)
  - No AI interference, pure time-based phase cycling
  - tripinfos.xml is automatically generated

Move output file (rename for clarity):
───────────────────────────────────────────────────────────────────────────

    # After baseline run completes:
    copy scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos.xml \\
         scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos_baseline.xml

   
Calculate Baseline Average Travel Time:
───────────────────────────────────────────────────────────────────────────

    python evaluate_metrics.py scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos_baseline.xml

Example output:
    📊 Trip Metrics:
      Total trips: 1245
      ...
      ⏱️  Average Travel Time: 52.34s

✅ RECORD THIS NUMBER: 52.34s (BASELINE ATT)


╔════════════════════════════════════════════════════════════════════════════╗
║ STEP 3: GENERATE COMPARISON PLOT                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

Create a side-by-side bar chart showing improvement:

Command:
───────────────────────────────────────────────────────────────────────────

    python plot_comparison.py \\
        scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos.xml \\
        scenarios/grid_3x3_lanes3_dynamic_dense/tripinfos_baseline.xml \\
        comparison_results.png

Output:
  📊 comparison_results.png + console summary

Console output example:
    • STMARL ATT: 39.05s
    • Baseline ATT: 52.34s
    • Improvement: 25.4%
    • Time saved per trip: 13.29s

✅ INSERT THIS PLOT INTO YOUR THESIS!


╔════════════════════════════════════════════════════════════════════════════╗
║ BONUS: Green Wave Visualization (Figure 8 equivalent)                     ║
╚════════════════════════════════════════════════════════════════════════════╝

To prove AI creates coordinated "Green Waves":

Option A: Quick visual proof (GUI observation)
───────────────────────────────────────────────────────────────────────────
  1. Run: python train_gnn_lstm_dqn.py ... --gui
  2. Watch how sequential intersections blink green in sequence
  3. Record a 30-second video screen capture
  4. Include video/screenshots in thesis appendix

Option B: Quantitative Green Wave plot (Advanced)
───────────────────────────────────────────────────────────────────────────
  1. Modify train_gnn_lstm_dqn.py to log phase changes (see plot_green_wave.py)
  2. Save phase history to CSV during test run
  3. python plot_green_wave.py phase_changes.csv
  4. Generates heatmap showing phase coordination over time

📖 Reference: Reproduces Figure 8 from STMARL paper (Phase ID vs Time)


╔════════════════════════════════════════════════════════════════════════════╗
║ THESIS REPORT STRUCTURE                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Section: "Experimental Results"

1. Average Travel Time Comparison (Metric)
   - Bar chart: STMARL vs Fixed-time
   - Table: ATT values + improvement percentage
   - Interpretation: AI saves X% time per vehicle

2. Coordination Analysis (Qualitative)
   - Video/screenshot: Green Waves in action
   - Phase plot: Shows AI learns to sync signals (if available)
   - Explanation: How spatio-temporal LSTM enables coordination

3. Statistical Significance (Optional)
   - Run 5-10 independent tests
   - Report mean ATT ± std dev for both methods
   - Show improvement is consistent, not random

Expected result for good STMARL:
  ✅ 20-30% reduction in ATT vs baseline
  ✅ Visible Green Wave patterns in GUI
  ✅ P < 0.05 if you run statistical tests


╔════════════════════════════════════════════════════════════════════════════╗
║ QUICK CHECKLIST                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Before writing thesis:
  ☐ Run STMARL test, get tripinfos.xml, calculate ATT
  ☐ Run baseline fixed-time, get tripinfos_baseline.xml, calculate ATT
  ☐ Generate comparison plot
  ☐ (Optional) Record GUI video of Green Waves
  ☐ (Optional) Generate phase coordination plot
  ☐ Copy plots to thesis Word document
  ☐ Write interpretation paragraph

Key numbers to include:
  📊 ATT improvement: (Baseline - STMARL) / Baseline * 100 %
  ⏱️  ATT STMARL: X.XX seconds
  ⏱️  ATT Baseline: Y.YY seconds
  🚗 Total vehicles tested: N


════════════════════════════════════════════════════════════════════════════
Questions? Your code is 100% complete. Now just: RUN → MEASURE → REPORT! 🎉
════════════════════════════════════════════════════════════════════════════
""")
