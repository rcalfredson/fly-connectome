"""Plot saved matrices independently of extraction and authentication."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def order_matrix(matrix, row_order=None, column_order=None):
    """Accept full permutations of body IDs without changing extracted data."""
    rows = sorted(matrix.index) if row_order is None else list(row_order)
    columns = sorted(matrix.columns) if column_order is None else list(column_order)
    for requested, original in [(rows, matrix.index), (columns, matrix.columns)]:
        if len(requested) != len(original) or len(set(requested)) != len(requested) or set(requested) != set(original):
            raise ValueError("Plot order must be a complete permutation of the axis body IDs.")
    return matrix.loc[rows, columns]


def load_matrix(path):
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype("int64")
    matrix.columns = matrix.columns.astype("int64")
    return matrix


def plot_heatmap(matrix, output_prefix, source_type, target_type, *, row_order=None, column_order=None,
                 row_labels=None, column_labels=None, row_groups=None, column_groups=None,
                 ordering_label=None, note=None):
    """Plot any explicit ordering; labels/groups are body-ID keyed and source agnostic."""
    matrix = order_matrix(matrix, row_order, column_order)
    height = max(4, 1.8 + 0.23 * len(matrix.index))
    width = max(6, 2.2 + 0.23 * len(matrix.columns)) + (4 if row_labels else 0)
    height += 3 if column_labels else 0
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    try:
        image = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0,
                          vmax=max(1, int(matrix.to_numpy().max())), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(matrix.columns)), labels=[column_labels[v] if column_labels else str(v)
                                                        for v in matrix.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(len(matrix.index)), labels=[row_labels[v] if row_labels else str(v)
                                                      for v in matrix.index], fontsize=8)
        for ids, groups, line in [(matrix.index, row_groups, ax.axhline),
                                  (matrix.columns, column_groups, ax.axvline)]:
            if groups is not None:
                for i in range(1, len(ids)):
                    if groups[ids[i]] != groups[ids[i - 1]]:
                        line(i - 0.5, color="white", linewidth=0.65, alpha=0.8)
        ax.set_xlabel(f"Postsynaptic {target_type} body ID")
        ax.set_ylabel(f"Presynaptic {source_type} body ID")
        title = f"{source_type} → {target_type} | raw connectivity"
        if ordering_label:
            title += f" | {ordering_label}"
        ax.set_title(title)
        if note:
            fig.suptitle(note, fontsize=9)
        fig.colorbar(image, ax=ax, label="Synapse count")
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for extension in ["png", "pdf"]:
            fig.savefig(str(prefix) + f".{extension}", dpi=180)
    finally:
        plt.close(fig)
