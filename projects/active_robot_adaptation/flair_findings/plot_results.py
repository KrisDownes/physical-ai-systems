"""Render the retained FLAIR analysis summaries."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "flair_pilot"
archived = json.loads((RESULTS / "author_data_analysis_summary.json").read_text())
local = json.loads((RESULTS / "author_three_condition_pilot.json").read_text())
published = archived["analyses"][0]
keys = ["driver_nominal", "driver", "flair"]
labels = ["Nominal\ndriver", "Damaged\ndriver", "Damaged\nFLAIR"]
x = np.arange(len(keys))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
for ax, metric, label in zip(axes, ["tracking_errors", "completion_times"],
                             ["Tracking error", "Completion (timestamp seconds)"]):
    a = published[metric]["medians"]
    b = local["metrics"][metric]["values"]
    ax.bar(x - .18, [a[k] for k in keys], .36, label="Archived: median of 20", color="#3769a5")
    ax.bar(x + .18, [b[k] for k in keys], .36, label="Local: one run", color="#e18a38")
    ax.set_xticks(x, labels)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8)
fig.suptitle("Hexapod results under the authors' analysis")
fig.savefig(HERE / "results.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
values = [entry["tracking_errors"]["recovered_overhead_percent"] for entry in archived["analyses"]]
bars = ax.bar(["Author conversion", "Raw angular rate", "Negated angular rate"], values,
              color=["#3769a5", "#e18a38", "#e18a38"])
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_ylim(0, max(values) * 1.2)
ax.set_ylabel("Recovered error overhead (%)")
ax.set_title("Sensitivity to angular-rate conversion: archived data")
ax.grid(axis="y", alpha=.2)
ax.set_axisbelow(True)
fig.savefig(HERE / "angular_sensitivity.png", dpi=170)
