"""Figure construction for the IMRI project.

Every module here is a **consumer** of scientific truth, never a source of it.
Plot code may select, aggregate and format results, but it must not define a
physical prescription, a threshold, or a criterion: those come from
:mod:`physical_process`, :mod:`disk_quantity`, :mod:`lisa` and
:mod:`workflows`.

Modules
-------
``rate_plots``
    Loading, validating and plotting single-halo rate / time / status results.
``disk_plots``
    Disk structure, pAGN-vs-in-repo solver comparison, and torque-component
    figures.
``batch_plots``
    Merger-tree batch interface figures (selected-host domain, job status).
``opacity_plots``
    Opacity-table interpolation verification figure.

All modules select the non-interactive ``Agg`` backend, so importing them is
safe inside worker processes and headless environments.
"""

__all__ = ["batch_plots", "disk_plots", "opacity_plots", "rate_plots"]
