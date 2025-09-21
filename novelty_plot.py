import matplotlib.pyplot as plt
import numpy as np

def plot_all_novelties_together(novelties, baseline_means, dropout_means,
                                baseline_stds, dropout_stds,
                                out_path="plot.pdf"):
    num = len(novelties)
    x = np.arange(num)  # positions for novelties
    width = 0.45        # bar width

    base_fs = 32
    plt.rcParams["font.family"] = "Times New Roman"

    fig, ax = plt.subplots(figsize=(8*num, 9))

    # Baseline and Dropout bars
    bars1 = ax.bar(x - width/2, baseline_means, width, label="Baseline", color="steelblue")
    bars2 = ax.bar(x + width/2, dropout_means, width, label="Confident Representation", color="firebrick")

    # Novelty names on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(novelties, fontsize=base_fs)
    ax.tick_params(axis='x', pad=20)

    # Shared y-axis scale across all novelties
    ymin = min(min(baseline_means), min(dropout_means)) - 0.2
    ymax = max(max(baseline_means), max(dropout_means)) + 0.2
    ax.set_ylim(ymin, ymax)

    y_ticks = np.arange(np.floor(ymin), np.ceil(ymax)+0.1, 0.5)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=base_fs)
    
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(3)
    
    ax.tick_params(axis='both', colors="black", width=3)

    # Add value labels with std above bars
    for i, bar in enumerate(bars1):
        mean, std = baseline_means[i], baseline_stds[i]
        ax.annotate(f"{mean:.2f}±{std:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, mean),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=base_fs)

    for i, bar in enumerate(bars2):
        mean, std = dropout_means[i], dropout_stds[i]
        ax.annotate(f"{mean:.2f}±{std:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, mean),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=base_fs)

    # Legend at bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=base_fs, frameon=False)

    plt.subplots_adjust(top=0.95, bottom=0.2)

    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


# Example data
novelties = ["Hidden Enemy", "Deceptive Enemy", "Novel Colors", "Invert Health"]
baseline_means = [2.8, 2.35, 2.7667, 2.9387]
dropout_means = [3.0714, 3.225, 2.9667, 3.1333]
baseline_stds  = [1.5948, 1.5877, 1.7638, 0.8071]
dropout_stds   = [1.4439, 1.7275, 1.3597, 1.1101]

plot_all_novelties_together(novelties, baseline_means, dropout_means,
                            baseline_stds, dropout_stds)
