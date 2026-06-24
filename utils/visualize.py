import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


def random_crop_spatial_adata(adata, window_size=16, x_start=None, y_start=None):
    """
    Crop a spatial window from AnnData based on spatial coordinates.
    """
    spatial_coords = adata.obsm["spatial"].copy()
    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]

    x_min_total = x_coords.min()
    x_max_total = x_coords.max()
    y_min_total = y_coords.min()
    y_max_total = y_coords.max()

    x_start_max = x_max_total - window_size
    y_start_max = y_max_total - window_size

    if x_start_max < x_min_total or y_start_max < y_min_total:
        rel_x = x_coords - x_min_total
        rel_y = y_coords - y_min_total
        norm_x = (rel_x / max(x_max_total - x_min_total, 1) * (window_size - 1)).round().astype(int)
        norm_y = (rel_y / max(y_max_total - y_min_total, 1) * (window_size - 1)).round().astype(int)
        adata.obsm["spatial"] = np.column_stack([norm_x, norm_y])
        return adata, {"x_min": x_min_total, "x_max": x_max_total, "y_min": y_min_total, "y_max": y_max_total}

    if x_start is None:
        x_start = np.random.uniform(x_min_total, x_start_max)
    else:
        x_start = np.clip(x_start, x_min_total, x_start_max)

    if y_start is None:
        y_start = np.random.uniform(y_min_total, y_start_max)
    else:
        y_start = np.clip(y_start, y_min_total, y_start_max)

    x_end = x_start + window_size
    y_end = y_start + window_size

    in_window = (
        (x_coords >= x_start) & (x_coords <= x_end) &
        (y_coords >= y_start) & (y_coords <= y_end)
    )
    cropped_adata = adata[in_window, :].copy()

    crop_bounds = {
        "x_min": x_start,
        "x_max": x_end,
        "y_min": y_start,
        "y_max": y_end,
    }
    return cropped_adata, crop_bounds


def plot_spatial_by_label(
    adata,
    label_key,
    spot_size=1,
    title=None,
    palette=None,
    legend_loc="right margin",
    save_path=None,
):
    axes = sc.pl.spatial(
        adata,
        color=label_key,
        spot_size=spot_size,
        img_key=None,
        title=title,
        frameon=False,
        show=False,
        palette=palette,
        legend_loc=legend_loc,
    )

    if isinstance(axes, list):
        for ax in axes:
            ax.patch.set_alpha(0.0)
    else:
        axes.patch.set_alpha(0.0)

    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches="tight", dpi=300)
        plt.close()

    return axes
