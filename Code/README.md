# Arctic Settlement Maritime Connection Opportunities

Analysis code for *Sea-ice retreat reshapes the spatial pattern of maritime connection opportunities among Arctic settlements*.

## Contents

- `Data_Preparation`: settlement inventories, coastal selection and port attributes.
- `Spatial_Framework`: land masks and settlement-to-departure mappings.
- `Preprocessing`: PIOMAS interpolation to 5 km and correction of inconsistent concentration/thickness values.
- `Connectivity`: POLARIS passability, daily connected components, annual and monthly summaries.
- `Regional_and_Ship_Class_Analysis`: regional assignment and regional/ship-class disparities.
- `Port_Coverage`: endpoint coverage and minimum additional port locations under the 100 km land-connectivity rule. `prepare_land_connectivity.py` generates the land-component assignments reused by the 100 km calculation.
- `Figures`: study-area, sea-ice, connectivity, disparity and port maps.
- `Parameters`: the original preprocessing configuration and standard POLARIS risk-index table.

## Setup and use

The original workflow uses Windows and Python. Install the packages in `requirements.txt`; SciPy must include `scipy.optimize.milp` (HiGHS). Times New Roman is used for manuscript figures.

```text
python -m pip install -r requirements.txt
python restore_project_layout.py --target Reproduction
```

The English folders organize the code for review. The restoration command recreates the original script names and relative paths in a new working directory; use those restored scripts for execution. `file_manifest.csv` maps every packaged file to its original project path and records its SHA-256 checksum. The scientific scripts and model settings are unchanged.

Run stages in this order: settlement preparation and spatial framework; PIOMAS preprocessing and correction; daily connectivity (`run_r2.py`) and summaries (`summarize_r2.py`); regional assignment (`assign.py`), regional analysis (`analyze.py`) and summaries (`summarize.py`); port land assignments, 100 km coverage and minimum-cover optimization; figures. Some spatial scripts represent successive baseline-building steps and consume the preceding step's output. Run scripts from the restored project root and review their input/output paths before execution.

## Data inputs

This is a code-only package. Obtain the public inputs from the sources below and use the relative paths specified in the scripts. The analysis period is 1979–2025, with 4,258 coastal settlement records, 1,248 departure cells and 12 ship classes. PIOMAS daily records use a 365-day calendar. The preprocessing script also supports the partial 2026 source files in the original download; the reported connectivity analysis ends in 2025.

- PIOMAS v2.1: https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid/ (daily concentration `aiday`, effective thickness `hiday`, and native grid files).
- AMAP boundary: https://www.amap.no/about/geographical-coverage
- GSHHG coastlines: https://www.ngdc.noaa.gov/mgg/shorelines/gshhs.html
- LME boundaries: https://github.com/datasets/lme-large-marine-ecosystems/tree/main/polygons/shp
- GeoNames: https://download.geonames.org/export/dump/
- Russian settlements: https://github.com/NickyX3/russia_settlements
- Swedish settlement layer: https://www.arcgis.com/home/item.html?id=23dbc7de31474b50a2cb47ab0cec53c8
- Russian port sources: https://base.garant.ru/6742107/ ; https://base.garant.ru/70581512/ ; https://www.rosmorport.ru/services/seaports/
- Other national settlement and port sources are listed in the manuscript's Open Research section and References. Natural Earth background maps: https://www.naturalearthdata.com/

Principal generated inputs are the fixed settlement/departure CSV tables, land and AMAP masks, corrected monthly 5 km NetCDF files, daily component-label NPZ files and regional assignment tables. These are generated or assembled by the project workflow and are not bundled here. The package does not automatically download all inputs or execute the full analysis.

## Checks

Packaged Python files were syntax-checked and their checksums matched the source files. The scientific calculations were not rerun when preparing this archive.
