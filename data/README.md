# Data

Raw data are **not** stored in this repository.

1. Go to the BGS dataset page: https://doi.org/10.5285/9c324699-abf1-4412-ab9f-39308507082d
2. Download the *UK Digital Geothermal Catalogue v2* archive.
3. Unzip it anywhere inside `data/raw/`. The script searches for `DGC_V2_GB_TEMP_DATA.xlsx` automatically.

Expected files include:

```
DGC_V2_GB_TEMP_DATA.xlsx      # temperature observations (model target)
DGC_V2_GB_HF_DATA.xlsx        # heat flow (context only)
DGC_V2_GB_COND_DATA.xlsx      # thermal conductivity
DGC_V2_SITE_META_DATA.xlsx    # site metadata
```

Running `python src/analyze.py` writes the cleaned tables to `data/processed/` (also not tracked).
