"""FIG. 9 for the Sannidhi provisional specification: black-and-white results figure.
Run:  python make_fig9.py        (needs matplotlib)
Output: FIG9_results.png (300 dpi) and FIG9_results.svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})

models = ["MLP", "CNN", "GRU", "LSTM", "Transformer", "MobileNetV3", "FallGuardNet\nfirst version", "FallGuardNet"]
f1 = [0.455, 0.562, 0.476, 0.321, 0.667, 0.750, 0.667, 0.889]
spec = [0.883, 0.783, 0.900, 0.383, 0.867, 0.917, 0.867, 1.000]

fig = plt.figure(figsize=(8.0, 10.6), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.05, 1.05], hspace=0.55)

# (a) confusion matrix
ax = fig.add_subplot(gs[0])
cm = [[60, 0], [2, 8]]
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
for r in range(2):
    for c in range(2):
        ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, fill=False, ec="black", lw=1.5))
        ax.text(c + 0.5, 1.5 - r, str(cm[r][c]), ha="center", va="center", fontsize=22, fontweight="bold")
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["Predicted\nnon-fall", "Predicted\nfall"])
ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["Actual\nnon-fall", "Actual\nfall"])
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_aspect("equal")
ax.text(-0.5, -0.35, "(a)", fontsize=12, fontweight="bold", transform=ax.transData)

def bars(ax, vals, label, tag):
    hatches = ["", "//", "\\\\", "xx", "..", "++", "--", ""]
    x = range(len(models))
    for i, v in enumerate(vals):
        last = i == len(vals) - 1
        ax.bar(i, v, width=0.7, fill=last, facecolor="black" if last else "white", edgecolor="black",
               hatch=None if last else hatches[i % len(hatches)], lw=1.2)
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("\n", " ") for m in models], fontsize=9, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.12); ax.set_ylabel(label)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.05, 1.07, tag, transform=ax.transAxes, fontsize=12, fontweight="bold")

bars(fig.add_subplot(gs[1]), f1, "F1 score", "(b)")
bars(fig.add_subplot(gs[2]), spec, "Specificity", "(c)")

fig.text(0.5, -0.035, "FIG. 9", ha="center", fontsize=14, fontweight="bold")
fig.savefig("FIG9_results.png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig("FIG9_results.svg", facecolor="white", bbox_inches="tight")
print("saved")