# Documentation

**Scientific overview:** [Gas-assisted inspirals in high-redshift galactic nuclei](../../articles/imri_pre/main.tex) is now a modular AASTeX 7 manuscript in `~/Phy/articles/imri_pre/`, with a compiled [PDF](../../articles/imri_pre/main.pdf). It covers the abstract, introduction, model, methods, results, discussion, limitations and conclusions. Edit the LaTeX section files there; the original Markdown is archived there under `provenance/`.

Four supporting guides, each with a separate purpose:

| Guide | Contents |
|---|---|
| [Run and workflow](quickstart.md) | Setup, CLI, execution stages, output schemas and validation |
| [Physical mechanisms](physical_mechanisms.md) | LaTeX equations, model assumptions, implementation ownership and completed capabilities |
| [Population prescriptions](published_populations.md) | IMF/component normalization, radial sampling, literature alignment and population-control commands |
| [Results and uncertainties](simulation_results_synthesis.md) | All saved experiment conclusions, quantitative comparisons, caveats, reproduction commands and next priorities |

Suggested reading: mechanisms → populations → results; use the run guide when executing simulations. The default population is `legacy_popiii`; Pan/Xue presets are opt-in controls. Detailed experiment CSVs, audit reports and figure notes remain under `outputs/`, rather than being duplicated across these guides. The [output index](../outputs/README.md) maps core results, supporting studies and archives, including their dependencies.
