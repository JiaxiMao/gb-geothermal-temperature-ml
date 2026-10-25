"""Predict subsurface temperature in Great Britain from the BGS Digital Geothermal Catalogue v2.

Pipeline
--------
1. Load and clean the GB temperature, heat-flow, conductivity and site tables.
2. Keep equilibrium temperature observations as the modelling target.
3. Hold out whole 50 km British National Grid blocks for testing.
4. Compare two baselines (median, linear geothermal gradient) with
   random forest and histogram gradient boosting.
5. Save metrics, row-level holdout predictions and figures.

Run from the repository root:

    python src/analyze.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"

RANDOM_STATE = 42
BLOCK_SIZE_M = 50_000  # spatial holdout block size (British National Grid)
TEST_FRACTION = 0.2

EQUILIBRIUM_TYPES = {
    "Equilibrium Temperature Log Data",
    "Equilibrium Temperature Log Data (Open Hole)",
    "Equilibrium Measurement",
}

# Identifiers, corrected temperatures, gradients, conductivity and heat flow
# are deliberately excluded to avoid target leakage.
FEATURES = [
    "DEPTH_M",
    "X",
    "Y",
    "EST_MEAN_ANNUAL_GROUND_TEMP",
    "GROUND_LEVEL",
    "RT_ELEVATION",
    "DRILLED_LENGTH",
]
TARGET = "TEMP"

# Approximate British National Grid rectangle around Cornwall.
# Not an official boundary: used only as a coverage check.
CORNWALL_X = (140_000, 270_000)
CORNWALL_Y = (30_000, 120_000)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def find_archive() -> Path:
    """Locate the folder that holds the BGS catalogue spreadsheets."""
    matches = list(RAW.rglob("DGC_V2_GB_TEMP_DATA.xlsx"))
    if not matches:
        raise FileNotFoundError(
            "BGS catalogue not found. Download it into data/raw/ "
            "(see data/README.md)."
        )
    return matches[0].parent


def read_table(archive: Path, name: str) -> pd.DataFrame:
    """Read a BGS sheet, drop its units row and normalise missing-value codes."""
    df = pd.read_excel(archive / name)
    # Each BGS table has a first row of units such as "M" or "degC".
    if "GEOTHERM_ID" in df:
        df = df[pd.to_numeric(df["GEOTHERM_ID"], errors="coerce").notna()].copy()
    df.columns = [str(c).strip() for c in df.columns]
    placeholders = {"Not Entered": np.nan, "Not Applicable": np.nan, "?": np.nan, "": np.nan}
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(placeholders)
        converted = pd.to_numeric(df[col], errors="coerce")
        # Treat a column as numeric when at least 80% of its values parse.
        if converted.notna().sum() >= max(3, int(df[col].notna().sum() * 0.8)):
            df[col] = converted
            df.loc[df[col] == -9999, col] = np.nan
    return df


def clean_temperature(temp: pd.DataFrame) -> pd.DataFrame:
    numeric = [TARGET, *FEATURES]
    for col in numeric:
        if col in temp:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").replace(-9999, np.nan)
    temp["GEOTHERM_ID"] = pd.to_numeric(temp["GEOTHERM_ID"], errors="coerce")
    temp = temp.dropna(subset=["GEOTHERM_ID", TARGET, "DEPTH_M", "X", "Y"]).copy()
    temp = temp[temp[TARGET].between(0, 155) & temp["DEPTH_M"].between(0, 10_504)]
    temp["GEOTHERM_ID"] = temp["GEOTHERM_ID"].astype("int64")
    return temp


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
def spatial_blocks(df: pd.DataFrame) -> pd.Series:
    bx = (df["X"].astype(float) // BLOCK_SIZE_M).astype(int).astype(str)
    by = (df["Y"].astype(float) // BLOCK_SIZE_M).astype(int).astype(str)
    return bx + "_" + by


def build_models() -> dict:
    """Each model is (estimator, feature list). All share median imputation."""
    return {
        "Median baseline": (DummyRegressor(strategy="median"), FEATURES),
        "Linear gradient baseline (depth only)": (LinearRegression(), ["DEPTH_M"]),
        "Random forest": (
            RandomForestRegressor(
                n_estimators=250, min_samples_leaf=3, max_features=0.9,
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            FEATURES,
        ),
        "Histogram gradient boosting": (
            HistGradientBoostingRegressor(
                max_iter=250, max_leaf_nodes=20, l2_regularization=1.0,
                learning_rate=0.08, random_state=RANDOM_STATE,
            ),
            FEATURES,
        ),
    }


def evaluate(primary: pd.DataFrame):
    X = primary[FEATURES].replace([np.inf, -np.inf], np.nan)
    y = primary[TARGET].astype(float)
    blocks = spatial_blocks(primary)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=blocks))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rows, predictions = [], {}
    for name, (estimator, cols) in build_models().items():
        pipeline = make_pipeline(SimpleImputer(strategy="median"), estimator)
        pipeline.fit(X.iloc[train_idx][cols], y_train)
        pred = pipeline.predict(X.iloc[test_idx][cols])
        predictions[name] = pred
        row = {
            "model": name,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_spatial_blocks_train": blocks.iloc[train_idx].nunique(),
            "n_spatial_blocks_test": blocks.iloc[test_idx].nunique(),
            "MAE_C": mean_absolute_error(y_test, pred),
            "RMSE_C": math.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        }
        if name.startswith("Linear gradient"):
            row["gradient_C_per_km"] = pipeline[-1].coef_[0] * 1000
        rows.append(row)

    metrics = pd.DataFrame(rows).sort_values("MAE_C").reset_index(drop=True)
    return metrics, predictions, train_idx, test_idx


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180)
    plt.close(fig)


def make_figures(temp, primary, hf, metrics, best, y_test, best_pred) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = min(80, float(temp[TARGET].quantile(0.98)))
    s = ax.scatter(temp["X"] / 1000, temp["Y"] / 1000, c=temp[TARGET], s=8, alpha=0.55,
                   cmap="inferno", vmin=0, vmax=vmax)
    fig.colorbar(s, ax=ax, label="Observed temperature (°C)")
    ax.set(title="BGS Great Britain subsurface temperature observations",
           xlabel="Easting (km, EPSG:27700)", ylabel="Northing (km, EPSG:27700)")
    save(fig, "gb_temperature_observations.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(primary["DEPTH_M"], primary[TARGET], s=8, alpha=0.28, c="#22577a", edgecolors="none")
    ax.set(title="Equilibrium temperature observations by reported depth",
           xlabel="Reported depth (m)", ylabel="Observed temperature (°C)")
    save(fig, "temperature_vs_depth.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_pred, s=13, alpha=0.42, c="#2a9d8f", edgecolors="none")
    lo = float(min(y_test.min(), best_pred.min()))
    hi = float(max(y_test.max(), best_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1)
    ax.set(title=f"Spatial holdout: {best}",
           xlabel="Observed temperature (°C)", ylabel="Predicted temperature (°C)")
    save(fig, "spatial_holdout_predicted_vs_observed.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    m = metrics.sort_values("MAE_C")
    colours = ["#2a9d8f" if n == best else "#9aa0a6" for n in m["model"]]
    ax.barh(m["model"], m["MAE_C"], color=colours)
    ax.invert_yaxis()
    ax.set(title="Spatially held-out model comparison", xlabel="Mean absolute error (°C)")
    save(fig, "model_comparison_mae.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(hf):
        s = ax.scatter(hf["X"] / 1000, hf["Y"] / 1000, c=hf["HF_VALUE"], s=28, cmap="magma",
                       edgecolor="black", linewidth=0.25)
        fig.colorbar(s, ax=ax, label="Reported heat flow (mW/m²)")
    ax.set(title="Historical BGS heat-flow observations (context only)",
           xlabel="Easting (km, EPSG:27700)", ylabel="Northing (km, EPSG:27700)")
    save(fig, "gb_heat_flow_observations.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for p in (PROCESSED, METRICS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)

    archive = find_archive()
    temp = clean_temperature(read_table(archive, "DGC_V2_GB_TEMP_DATA.xlsx"))
    site = read_table(archive, "DGC_V2_SITE_META_DATA.xlsx")
    cond = read_table(archive, "DGC_V2_GB_COND_DATA.xlsx")
    hf = read_table(archive, "DGC_V2_GB_HF_DATA.xlsx")

    hf["HF_VALUE"] = pd.to_numeric(hf.get("HF_VALUE"), errors="coerce").replace(-9999, np.nan)
    hf = hf.dropna(subset=["HF_VALUE", "X", "Y"]).copy()

    # Cleaned tables (regenerated locally; not committed to git).
    temp.to_csv(PROCESSED / "gb_temperature_clean.csv", index=False)
    hf.to_csv(PROCESSED / "gb_heat_flow_clean.csv", index=False)
    cond.to_csv(PROCESSED / "gb_conductivity_clean.csv", index=False)
    site.to_csv(PROCESSED / "site_metadata_clean.csv", index=False)

    primary = temp[temp["TEMP_OBS_TYPE"].astype(str).isin(EQUILIBRIUM_TYPES)].copy()
    if len(primary) < 100:
        raise ValueError(f"Only {len(primary)} equilibrium observations; check the quality filter.")

    metrics, predictions, train_idx, test_idx = evaluate(primary)
    metrics.to_csv(METRICS / "model_metrics.csv", index=False)

    baselines = metrics["model"].str.contains("baseline")
    best = metrics.loc[~baselines, "model"].iloc[0]
    best_pred = predictions[best]
    y_test = primary[TARGET].astype(float).iloc[test_idx]

    holdout = primary.iloc[test_idx][["GEOTHERM_ID", "X", "Y", "DEPTH_M", TARGET, "TEMP_OBS_TYPE"]].copy()
    holdout["PREDICTED_TEMP_C"] = best_pred
    holdout["RESIDUAL_C"] = holdout[TARGET] - holdout["PREDICTED_TEMP_C"]
    holdout.to_csv(METRICS / "spatial_holdout_predictions.csv", index=False)

    make_figures(temp, primary, hf, metrics, best, y_test, best_pred)

    cornwall = temp[temp["X"].between(*CORNWALL_X) & temp["Y"].between(*CORNWALL_Y)]
    summary = {
        "source": "BGS UK Digital Geothermal Catalogue v2 (DOI 10.5285/9c324699-abf1-4412-ab9f-39308507082d)",
        "gb_temperature_rows_clean": int(len(temp)),
        "gb_temperature_unique_sites": int(temp["GEOTHERM_ID"].nunique()),
        "equilibrium_rows_modelled": int(len(primary)),
        "equilibrium_unique_sites": int(primary["GEOTHERM_ID"].nunique()),
        "holdout_unique_sites": int(primary.iloc[test_idx]["GEOTHERM_ID"].nunique()),
        "temperature_obs_type_counts": temp["TEMP_OBS_TYPE"].fillna("Unknown").value_counts().to_dict(),
        "gb_heat_flow_rows_clean": int(len(hf)),
        "gb_conductivity_rows_clean": int(len(cond)),
        "approximate_cornwall_window_rows": int(len(cornwall)),
        "approximate_cornwall_window_unique_sites": int(cornwall["GEOTHERM_ID"].nunique()),
        "spatial_block_size_m": BLOCK_SIZE_M,
        "best_model_by_mae": str(best),
        "metrics": metrics.to_dict(orient="records"),
    }
    (METRICS / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest model: {best}")


if __name__ == "__main__":
    main()
