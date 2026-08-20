# Figure scripts

The scripts in this directory generate the manuscript's schematic and analytical multi-panel figures. Shared layout, colors and export settings live in `style.py`; `FIGURE_CONTRACTS.md` records each figure's intended analytical claim.

`make_figure1.py` is schematic and can run after package installation. Figures 2-5 require the corresponding aggregate model or human-evaluation source-data tables. Restricted individual-level coach records are not bundled with this repository.

Run the modules from the repository root so that project-relative input paths resolve consistently, for example `python -m figures.make_figure1`. Generated PDF, SVG and raster files are written to ignored output or artifact directories.

Before journal deposition, source-data tables should be archived separately and their permanent accession recorded in the manuscript and release metadata. No DOI is invented in this repository.
