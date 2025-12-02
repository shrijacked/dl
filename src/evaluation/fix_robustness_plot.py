#!/usr/bin/env python3
"""
Fix the robustness ranking plot - the x-axis was starting at 80% 
but the corruption accuracy values are around 70-77%.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the data
data_path = Path("evaluation_outputs/reports/robustness_ranking.json")
with open(data_path) as f:
    data = json.load(f)

# Extract overall ranking data
overall = data["overall_ranking"]
models = [r["model"] for r in overall]
clean_acc = [r["clean_accuracy"] * 100 for r in overall]
corrupt_acc = [r["mean_corruption_accuracy"] * 100 for r in overall]

# Extract category rankings
categories = ["noise", "blur", "weather", "digital"]
category_colors = {
    "noise": "#636EFA",      # blue
    "blur": "#AB63FA",       # purple
    "weather": "#2CA02C",    # green (teal-ish)
    "digital": "#FFA15A"     # orange
}

# Build category accuracy matrix (model x category)
model_order = models  # same order as overall ranking
category_acc = {cat: {} for cat in categories}
for cat in categories:
    for entry in data["category_rankings"][cat]:
        category_acc[cat][entry["model"]] = entry["accuracy"] * 100

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- Left plot: Clean vs Corrupted Accuracy ---
y_pos = np.arange(len(models))
bar_height = 0.35

# Plot bars
bars_clean = ax1.barh(y_pos - bar_height/2, clean_acc, bar_height, 
                       label='Clean', color='#2ECC71', alpha=0.9)
bars_corrupt = ax1.barh(y_pos + bar_height/2, corrupt_acc, bar_height, 
                         label='Corrupted (Mean)', color='#E74C3C', alpha=0.9)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(models)
ax1.set_xlabel('Accuracy (%)')
ax1.set_title('Clean vs Corrupted Accuracy')
ax1.legend(loc='upper left')
ax1.set_xlim(65, 102)  # Fixed: Start from 65% to show corruption bars
ax1.invert_yaxis()  # Best model at top
ax1.grid(axis='x', alpha=0.3)

# --- Right plot: Robustness by Corruption Category ---
bar_width = 0.2
y_pos = np.arange(len(models))

for i, cat in enumerate(categories):
    acc_values = [category_acc[cat].get(m, 0) for m in models]
    offset = (i - 1.5) * bar_width
    ax2.barh(y_pos + offset, acc_values, bar_width, 
             label=cat, color=category_colors[cat], alpha=0.85)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(models)
ax2.set_xlabel('Accuracy (%)')
ax2.set_title('Robustness by Corruption Category')
ax2.legend(loc='lower right')
ax2.set_xlim(65, 85)  # Fixed: Range 65-85% to show all category bars
ax2.invert_yaxis()  # Same order as left plot
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()

# Save the fixed figure
output_path = Path("evaluation_outputs/figures/robustness_ranking.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved fixed plot to: {output_path}")

plt.show()

