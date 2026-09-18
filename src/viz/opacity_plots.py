"""Opacity-table verification figure.

Plots the tabulated Rosseland opacity grid against its interpolant so the
interpolation used by :mod:`disk_quantity.disk_solve` can be inspected.  The
physics lives in :mod:`disk_quantity.opacity`; this module only draws it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from disk_quantity.opacity import inter, kappa_data, logR_data, logT_data  # noqa: E402

FloatArray = npt.NDArray[np.float64]


def plot_opacity_verification(
    interpolator: RegularGridInterpolator,
    logR_vals: FloatArray,
    logT_vals: FloatArray,
    kappa_orig: FloatArray,
    logR_grid: FloatArray,
    logT_grid: FloatArray,
) -> Figure:
    """
    Plot the interpolated opacity over the full (logR, logT) domain.

    Parameters:
        interpolator : RegularGridInterpolator
            The opacity interpolator.
        logR_vals : array, shape (nR,)
            The original logR grid points.
        logT_vals : array, shape (nT,)
            The original logT grid points.
        kappa_orig : array, shape (nR, nT)
            The original opacity values (log10 kappa).
        logR_grid : array, shape (nR_fine,)
            Fine logR grid for plotting.
        logT_grid : array, shape (nT_fine,)
            Fine logT grid for plotting.
    """
    # Create fine mesh
    R_fine, T_fine = np.meshgrid(logR_grid, logT_grid, indexing="ij")
    points = np.column_stack((R_fine.ravel(), T_fine.ravel()))

    # Interpolate
    kappa_fine = interpolator(points).reshape(R_fine.shape)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original data (coarse)
    im1 = ax1.pcolormesh(
        logR_vals, logT_vals, kappa_orig.T, shading="auto", cmap="viridis"
    )
    ax1.set_xlabel("log R")
    ax1.set_ylabel("log T")
    ax1.set_title("Original data (coarse grid)")
    plt.colorbar(im1, ax=ax1, label="log κ")

    # Interpolated data (fine)
    im2 = ax2.pcolormesh(
        logR_grid, logT_grid, kappa_fine.T, shading="auto", cmap="viridis"
    )
    ax2.set_xlabel("log R")
    ax2.set_ylabel("log T")
    ax2.set_title("Interpolated (fine grid)")
    plt.colorbar(im2, ax=ax2, label="log κ")

    plt.tight_layout()
    pass  # caller decides how to display or save

    # Print some sample interpolated values
    print("Sample interpolation checks:")
    test_points = np.array([[-6.0, 3.5], [-3.0, 4.0], [0.0, 5.0], [-2.5, 4.5]])
    for pt in test_points:
        val = interpolator(pt)[0]
        print(f"  (logR={pt[0]:.2f}, logT={pt[1]:.2f}) -> logκ = {val:.4f}")
    return fig


def plot_default_opacity_verification(
    logR_count: int = 200,
    logT_count: int = 500,
) -> Figure:
    """Draw the verification figure for the canonical opacity table."""
    logR_fine = np.linspace(logR_data.min(), logR_data.max(), logR_count)
    logT_fine = np.linspace(logT_data.min(), logT_data.max(), logT_count)
    return plot_opacity_verification(
        inter, logR_data, logT_data, kappa_data, logR_fine, logT_fine
    )


__all__ = ["plot_default_opacity_verification", "plot_opacity_verification"]
