import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torch_geometric.datasets import TUDataset


def plot_histograms_for_multiple_matrices(
    datasets: List[np.ndarray],
    datasets_names: List[str],
    figname: str,
    bins=30,
    colors=None,
    alpha=0.7,
    figsize=(10, 6),
):
    """
    Plots histograms for multiple matrices, with each matrix represented by a different color.

    Parameters:
        matrices (list of numpy arrays): List of 2D arrays (matrices) to plot.
        labels (list of str): Labels for each matrix.
        bins (int or list): Number of bins or bin edges for the histograms.
        colors (list of str): Colors for each histogram.
        alpha (float): Transparency level of the histograms.
        figsize (tuple): Size of the figure.
    """
    num_datasets = len(datasets)

    if datasets_names is None:
        datasets_names = [f"Matrix {i+1}" for i in range(num_datasets)]

    if colors is None:
        colors = plt.cm.tab10.colors[:num_datasets]

    plt.figure(figsize=figsize)

    for matrix, dataset_name, color in zip(datasets, datasets_names, colors):
        # Flatten each matrix before plotting
        combined_data = matrix.flatten()
        # combined_data = np.clip(combined_data, -2, 2)
        plt.hist(
            combined_data,
            bins=bins,
            label=dataset_name,
            color=color,
            alpha=alpha,
            edgecolor="black",
        )

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Combined Histograms of Datasets")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"histograms_{figname}.png")
    print(f"Saved histograms to histograms_{figname}.png")


if __name__ == "__main__":
    dataset_names = ["PROTEINS", "ENZYMES", "MUTAG"]
    datasets = [
        TUDataset(root=f"data/pyg/TUDataset.{name}", name=name).x.numpy()
        for name in dataset_names
    ]
    plot_histograms_for_multiple_matrices(
        datasets,
        datasets_names=dataset_names,
        figname="GraphDatasets",
        bins=30,
    )
