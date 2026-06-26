import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(all_frames, n_clusters, spc, labels, cluster_out):
    fig, axes = plt.subplots(n_clusters, spc, figsize=(spc * 2, n_clusters * 2))

    if n_clusters == 1:
        return

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        n_avail = len(cluster_indices)
        chosen_pos = np.linspace(0, n_avail - 1, min(spc, n_avail), dtype=int)
        chosen = cluster_indices[chosen_pos]

        axes[cluster_id, 0].set_ylabel(f"C{cluster_id}", fontsize=7, rotation=0, labelpad=20)
        for col, idx in enumerate(chosen):
            ax = axes[cluster_id, col]
            ax.imshow(all_frames[idx])
            ax.axis("off")
        for col in range(len(chosen), spc):
            axes[cluster_id, col].axis("off")

    plt.suptitle("Sample frames per embedding cluster", fontsize=10)
    plt.tight_layout()
    plt.savefig(cluster_out, dpi=150)
    plt.show()