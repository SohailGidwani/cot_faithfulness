"""
Generates figures/faithfulness_summary.png — Figure 1 in the final report.
Averaged across Llama-3.2-3B and Qwen-2.5-7B.

Usage:
    python figures/make_summary_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({"font.size": 10, "font.family": "serif"})

metrics = ["SCR@1", "CFR (all)", "Max SBH"]
gsm8k = [(22.7 + 9.2) / 2,     # 15.95
         (60.3 + 48.4) / 2,    # 54.35
         (2.0  + 2.4) / 2]     # 2.2
arc   = [(59.3 + 82.7) / 2,    # 71.0
         (34.3 + 10.8) / 2,    # 22.55
         (17.2 + 16.8) / 2]    # 17.0

x = np.arange(len(metrics))
w = 0.38

fig, ax = plt.subplots(figsize=(4.2, 2.9))

b1 = ax.bar(x - w/2, gsm8k, w, label="GSM8K (Math)",
            color="#1f77b4", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w/2, arc,   w, label="ARC (Science MC)",
            color="#ff7f0e", edgecolor="black", linewidth=0.5)

ax.set_ylabel("Percentage (%)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 95)
ax.legend(loc="upper right", fontsize=8.5, frameon=True)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.3,
                "%.1f" % h, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "faithfulness_summary.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
print("Saved", out)
