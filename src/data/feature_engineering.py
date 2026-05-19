import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "Irrigation_Need"
ID_COL = "id"
EPS = 1e-6

BASE_CATEGORICAL_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]

BASE_NUMERICAL_COLS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

GROWTH_STAGE_MAP = {
    "Sowing": 0,
    "Vegetative": 1,
    "Flowering": 2,
    "Harvest": 3,
}

SEASON_MAP = {
    "Rabi": 0,
    "Zaid": 1,
    "Kharif": 2,
}

BASE_TARGET_AGG_KEYS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Region",
    "Soil_Crop_Combo",
    "Irrigation_Water_Combo",
    "Crop_Stage_Combo",
]

ANCHOR_NUMERIC_COLS = [
    "Soil_Moisture",
    "Rainfall_mm",
    "Previous_Irrigation_mm",
    "Temperature_C",
]

NUMERIC_AGG_PLAN = {
    "Soil_Type": {
        "cols": ["Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity", "Rainfall_mm"],
        "stats": ["mean", "std", "count", "nunique", "min", "max"],
    },
    "Crop_Type": {
        "cols": ["Soil_Moisture", "Temperature_C", "Humidity", "Previous_Irrigation_mm"],
        "stats": ["mean", "std", "count", "nunique", "min", "max"],
    },
    "Crop_Growth_Stage": {
        "cols": ["Soil_Moisture", "Temperature_C", "Humidity"],
        "stats": ["mean", "std", "count", "nunique", "min", "max"],
    },
    "Season": {
        "cols": ["Rainfall_mm", "Temperature_C", "Humidity", "Sunlight_Hours"],
        "stats": ["mean", "std", "count", "nunique", "median", "min", "max"],
    },
    "Irrigation_Type": {
        "cols": ["Previous_Irrigation_mm", "Soil_Moisture", "Field_Area_hectare"],
        "stats": ["mean", "std", "count", "nunique", "min", "max"],
    },
    "Water_Source": {
        "cols": ["Previous_Irrigation_mm", "Rainfall_mm", "Electrical_Conductivity"],
        "stats": ["mean", "std", "count", "nunique", "min", "max"],
    },
    "Region": {
        "cols": ["Rainfall_mm", "Temperature_C", "Humidity", "Wind_Speed_kmh"],
        "stats": ["mean", "std", "count", "nunique", "median", "min", "max"],
    },
    "Soil_Crop_Combo": {
        "cols": ["Soil_Moisture", "Rainfall_mm", "Previous_Irrigation_mm"],
        "stats": ["mean", "std", "count", "nunique"],
    },
    "Irrigation_Water_Combo": {
        "cols": ["Previous_Irrigation_mm", "Field_Area_hectare", "Rainfall_mm"],
        "stats": ["mean", "std", "count", "nunique"],
    },
    "Season_Region_Combo": {
        "cols": ["Rainfall_mm", "Temperature_C", "Humidity"],
        "stats": ["mean", "std", "count", "nunique", "median"],
    },
    "Mulch_Season_Combo": {
        "cols": ["Soil_Moisture", "Previous_Irrigation_mm", "Rainfall_mm"],
        "stats": ["mean", "std", "count", "nunique", "median"],
    },
}

GROUP_QUANTILES = [10, 25, 50, 75, 90]
GROUP_QUANTILE_PLAN = {
    "Season": ["Rainfall_mm", "Temperature_C"],
    "Region": ["Rainfall_mm", "Humidity"],
    "Crop_Type": ["Soil_Moisture", "Previous_Irrigation_mm"],
    "Soil_Type": ["Soil_Moisture", "Organic_Carbon"],
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan).add(EPS)


# Keep missing-category handling consistent before generating pairs, combos, or encodings.
def _categorical_key(series: pd.Series) -> pd.Series:
    return series.fillna("Missing").astype(str)


# Infer numeric columns dynamically so new engineered numeric features flow through automatically.
def _resolve_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col != ID_COL and pd.api.types.is_numeric_dtype(df[col])
    ]


# Everything that is not numeric and not the id is treated as categorical for downstream models.
def _resolve_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col != ID_COL and not pd.api.types.is_numeric_dtype(df[col])
    ]


# Impute on the combined frame so train and test see the same pre-target feature distribution.
def _fill_combined_frame(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = _resolve_numeric_columns(df)
    categorical_cols = _resolve_categorical_columns(df)

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = _categorical_key(df[col])

    return categorical_cols, numeric_cols


# Final cleanup uses train medians only, which keeps test imputation aligned with a real inference setup.
def _finalize_train_test_frames(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    categorical_cols = _resolve_categorical_columns(train_features)
    numerical_cols = _resolve_numeric_columns(train_features)

    for col in numerical_cols:
        train_median = train_features[col].median()
        train_features[col] = train_features[col].fillna(train_median)
        test_features[col] = test_features[col].fillna(train_median)

    for col in categorical_cols:
        train_features[col] = _categorical_key(train_features[col])
        test_features[col] = _categorical_key(test_features[col])

    return train_features, test_features, categorical_cols, numerical_cols


# Guard StratifiedKFold against small minority classes so feature generation does not fail on narrow folds.
def _resolve_stratified_splits(y: pd.Series, requested_splits: int) -> int:
    class_counts = y.value_counts()
    if class_counts.empty:
        return 2

    max_valid_splits = int(class_counts.min())
    return max(2, min(requested_splits, max_valid_splits))


def _quantile_band(
    series: pd.Series, labels: list[str], prefix: str
) -> pd.Series:
    # Rank first so qcut stays stable even when many values repeat.
    ranked = series.rank(method="first")
    bucket_count = min(len(labels), ranked.nunique())
    if bucket_count <= 1:
        return pd.Series([f"{prefix}_Single"] * len(series), index=series.index)

    selected_labels = labels[:bucket_count]
    return pd.qcut(
        ranked,
        q=bucket_count,
        labels=selected_labels,
        duplicates="drop",
    ).astype(str)


# Coarse bands help tree models learn threshold-like behavior from soil chemistry and weather levels.
def _add_bucket_features(df: pd.DataFrame) -> None:
    df["Soil_pH_Band"] = pd.cut(
        df["Soil_pH"],
        bins=[-np.inf, 5.5, 7.5, np.inf],
        labels=["Acidic", "Neutral", "Alkaline"],
    ).astype(str)
    df["Moisture_Band"] = _quantile_band(
        df["Soil_Moisture"],
        labels=["Very_Low", "Low", "High", "Very_High"],
        prefix="Moisture",
    )
    df["Rainfall_Band"] = _quantile_band(
        df["Rainfall_mm"],
        labels=["Low", "Medium", "High", "Very_High"],
        prefix="Rainfall",
    )
    df["Temperature_Band"] = _quantile_band(
        df["Temperature_C"],
        labels=["Cool", "Mild", "Warm", "Hot"],
        prefix="Temperature",
    )


# Pair direct sensor relationships that often correlate with irrigation demand.
def _add_interaction_features(df: pd.DataFrame) -> None:
    humidity_ratio = df["Humidity"] / 100.0
    sunlight_pressure = df["Temperature_C"] * df["Sunlight_Hours"]
    evaporation_pressure = sunlight_pressure * (1.0 - humidity_ratio + EPS)

    df["Moisture_to_Temperature"] = safe_divide(
        df["Soil_Moisture"], df["Temperature_C"]
    )
    df["Rainfall_to_Temperature"] = safe_divide(
        df["Rainfall_mm"], df["Temperature_C"]
    )
    df["Prev_Irrigation_to_Area"] = safe_divide(
        df["Previous_Irrigation_mm"], df["Field_Area_hectare"]
    )
    df["Rainfall_to_Area"] = safe_divide(
        df["Rainfall_mm"], df["Field_Area_hectare"]
    )
    df["Rain_plus_Irrigation"] = (
        df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    )
    df["Water_Availability_Index"] = safe_divide(
        df["Rain_plus_Irrigation"], df["Field_Area_hectare"]
    )
    df["Evaporation_Pressure"] = evaporation_pressure
    df["Dryness_Index"] = safe_divide(
        evaporation_pressure,
        df["Soil_Moisture"] + df["Previous_Irrigation_mm"],
    )
    df["Conductivity_pH_Interaction"] = (
        df["Electrical_Conductivity"] * df["Soil_pH"]
    )
    df["Carbon_Moisture_Interaction"] = (
        df["Organic_Carbon"] * df["Soil_Moisture"]
    )
    df["Wind_Sunlight_Interaction"] = (
        df["Wind_Speed_kmh"] * df["Sunlight_Hours"]
    )
    df["Temp_Humidity_Interaction"] = df["Temperature_C"] * humidity_ratio
    df["Rainfall_per_Humidity"] = safe_divide(df["Rainfall_mm"], df["Humidity"])
    df["Moisture_Deficit"] = 100.0 - df["Soil_Moisture"]
    df["Heat_Load"] = df["Temperature_C"] * (1.0 - humidity_ratio / 2.0)
    df["Irrigation_Gap"] = (
        df["Rainfall_mm"] - df["Previous_Irrigation_mm"]
    )


# Simple ordinal encodings let the model understand stage and season progression.
def _add_ordinal_features(df: pd.DataFrame) -> None:
    df["Crop_Growth_Stage_Ordinal"] = (
        df["Crop_Growth_Stage"].map(GROWTH_STAGE_MAP).fillna(-1).astype(int)
    )
    df["Season_Ordinal"] = df["Season"].map(SEASON_MAP).fillna(-1).astype(int)


# Approximate field-level water pressure by balancing incoming water against likely atmospheric demand.
def _add_irrigation_domain_features(df: pd.DataFrame) -> None:
    humidity_ratio = df["Humidity"] / 100.0
    effective_rainfall = df["Rainfall_mm"] * (
        1.0 - df["Wind_Speed_kmh"] / 150.0
    ).clip(lower=0.25)
    irrigation_supply = df["Previous_Irrigation_mm"] + effective_rainfall
    climate_demand = (
        df["Temperature_C"] * (1.0 + df["Sunlight_Hours"] / 12.0)
        + df["Wind_Speed_kmh"] * 0.35
        - df["Humidity"] * 0.08
    )

    df["Effective_Rainfall"] = effective_rainfall
    df["Irrigation_Supply"] = irrigation_supply
    df["Climate_Demand_Index"] = climate_demand
    df["Net_Water_Balance"] = irrigation_supply - climate_demand
    df["Water_Stress_Index"] = safe_divide(
        climate_demand,
        df["Soil_Moisture"] + irrigation_supply,
    )
    df["Moisture_Recharge_Ratio"] = safe_divide(
        irrigation_supply,
        df["Soil_Moisture"] + 1.0,
    )
    df["Field_Size_Stress"] = safe_divide(
        climate_demand * df["Field_Area_hectare"],
        irrigation_supply + 1.0,
    )
    df["Salinity_Stress_Index"] = (
        df["Electrical_Conductivity"] * (1.0 - humidity_ratio + EPS)
    )
    df["pH_Deviation"] = (df["Soil_pH"] - 6.5).abs()
    df["Soil_Buffer_Score"] = safe_divide(
        df["Organic_Carbon"],
        df["Electrical_Conductivity"] + df["pH_Deviation"] + 1.0,
    )
    df["Canopy_Demand_Index"] = (
        df["Crop_Growth_Stage_Ordinal"] + 1.0
    ) * climate_demand
    df["Seasonal_Water_Pressure"] = (
        (df["Season_Ordinal"] + 1.0) * df["Water_Stress_Index"]
    )
    df["High_Stress_Flag"] = (
        (df["Water_Stress_Index"] > 1.0) | (df["Net_Water_Balance"] < 0)
    ).astype(np.int8)
    df["Moisture_Critical_Flag"] = (
        (df["Soil_Moisture"] < 30.0) & (climate_demand > climate_demand.median())
    ).astype(np.int8)


# Crossed categories expose combinations such as crop-by-soil or irrigation-by-water source.
def _add_cross_features(df: pd.DataFrame) -> None:
    df["Soil_Crop_Combo"] = _categorical_key(df["Soil_Type"]) + "__" + _categorical_key(df["Crop_Type"])
    df["Season_Region_Combo"] = _categorical_key(df["Season"]) + "__" + _categorical_key(df["Region"])
    df["Irrigation_Water_Combo"] = (
        _categorical_key(df["Irrigation_Type"]) + "__" + _categorical_key(df["Water_Source"])
    )
    df["Crop_Stage_Combo"] = (
        _categorical_key(df["Crop_Type"]) + "__" + _categorical_key(df["Crop_Growth_Stage"])
    )
    df["Mulch_Season_Combo"] = (
        _categorical_key(df["Mulching_Used"]) + "__" + _categorical_key(df["Season"])
    )
    df["Soil_Season_Combo"] = (
        _categorical_key(df["Soil_Type"]) + "__" + _categorical_key(df["Season"])
    )
    df["Region_Water_Combo"] = (
        _categorical_key(df["Region"]) + "__" + _categorical_key(df["Water_Source"])
    )


# Dense pairwise categories can capture interactions that one-hot single columns may miss.
def _add_pairwise_categorical_features(df: pd.DataFrame) -> None:
    for left_idx, left_col in enumerate(BASE_CATEGORICAL_COLS[:-1]):
        for right_col in BASE_CATEGORICAL_COLS[left_idx + 1:]:
            pair_name = f"{left_col}_{right_col}_Pair"
            df[pair_name] = (
                _categorical_key(df[left_col]) + "__" + _categorical_key(df[right_col])
            )


# Missingness itself can be predictive when sensor availability or metadata quality varies by field.
def _add_missingness_features(df: pd.DataFrame) -> None:
    missing_code = np.zeros(len(df), dtype=np.float64)

    for idx, col in enumerate(BASE_CATEGORICAL_COLS):
        is_missing = df[col].isna().astype(np.float64)
        missing_code += is_missing * (2 ** idx)
        df[f"{col}_is_missing"] = is_missing

        for anchor_col in ANCHOR_NUMERIC_COLS:
            df[f"{col}_missing_{anchor_col}"] = is_missing * 1000.0 + df[anchor_col]

    df["Missing_Pattern_Code"] = missing_code
    df["Missing_Count"] = df.isna().sum(axis=1).astype(np.float64)


# Rounded versions help when source systems emit measurements at coarse resolutions.
def _add_rounding_features(df: pd.DataFrame) -> None:
    rounding_plan = {
        "Soil_Moisture": [0, 1],
        "Rainfall_mm": [0, 1],
        "Previous_Irrigation_mm": [0, 1],
        "Temperature_C": [0, 1],
    }

    for col, precisions in rounding_plan.items():
        for precision in precisions:
            df[f"{col}_round_{precision}"] = df[col].round(precision)


# Decimal-pattern features sometimes surface hidden device precision or manual-entry behavior.
def _add_digit_features(df: pd.DataFrame) -> None:
    digit_sources = {
        "Soil_Moisture": 3,
        "Rainfall_mm": 3,
        "Previous_Irrigation_mm": 3,
    }

    for col, max_digit in digit_sources.items():
        for digit_idx in range(1, max_digit + 1):
            feature_name = f"{col}_digit_{digit_idx}"
            digit_values = ((df[col] * (10 ** digit_idx)) % 10).fillna(-1)
            df[feature_name] = digit_values.astype(np.int16)

        first = f"{col}_digit_1"
        second = f"{col}_digit_2"
        df[f"{col}_digit_pair_1_2"] = (
            (df[first] + 1) * 11 + (df[second] + 1)
        ).astype(np.int16)


# Factorized category-plus-anchor combos create dense hybrid features for boosting models.
def _add_anchor_combo_features(df: pd.DataFrame) -> None:
    for cat_col in BASE_CATEGORICAL_COLS:
        factorized, _ = pd.factorize(_categorical_key(df[cat_col]), sort=True)
        factorized = pd.Series(factorized, index=df.index, dtype=np.int32)
        df[f"{cat_col}_factorized"] = factorized

        for anchor_col in ANCHOR_NUMERIC_COLS:
            df[f"{cat_col}_{anchor_col}_combo"] = (
                factorized.astype(np.float64) * 1000.0 + df[anchor_col]
            )


# Aggregate group statistics make each row aware of the historical context of its category bucket.
def _apply_groupby_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    for group_col, config in NUMERIC_AGG_PLAN.items():
        grouped = (
            enriched.groupby(group_col, dropna=False)[config["cols"]]
            .agg(config["stats"])
        )
        grouped.columns = [
            f"FE_{group_col}_{value_col}_{stat}"
            for value_col, stat in grouped.columns
        ]
        grouped = grouped.reset_index()
        enriched = enriched.merge(grouped, on=group_col, how="left")

    return enriched


# Group quantiles complement means by describing spread and skew inside each category.
def _apply_groupby_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    for group_col, value_cols in GROUP_QUANTILE_PLAN.items():
        for value_col in value_cols:
            quantile_frame = (
                enriched.groupby(group_col, dropna=False)[value_col]
                .quantile([q / 100.0 for q in GROUP_QUANTILES])
                .unstack()
            )
            quantile_frame.columns = [
                f"FE_{group_col}_{value_col}_q{quantile}"
                for quantile in GROUP_QUANTILES
            ]
            quantile_frame = quantile_frame.reset_index()
            enriched = enriched.merge(quantile_frame, on=group_col, how="left")

    return enriched


# Ratios derived from grouped stats often act like compact density and variability indicators.
def _add_groupby_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    for group_col, config in NUMERIC_AGG_PLAN.items():
        for value_col in config["cols"]:
            count_col = f"FE_{group_col}_{value_col}_count"
            nunique_col = f"FE_{group_col}_{value_col}_nunique"
            std_col = f"FE_{group_col}_{value_col}_std"

            if count_col in enriched.columns and nunique_col in enriched.columns:
                enriched[f"FE_{group_col}_{value_col}_count_per_nunique"] = safe_divide(
                    enriched[count_col],
                    enriched[nunique_col],
                )

            if std_col in enriched.columns and count_col in enriched.columns:
                enriched[f"FE_{group_col}_{value_col}_std_per_count"] = safe_divide(
                    enriched[std_col],
                    enriched[count_col],
                )

    return enriched


# Limit target encoding to curated high-signal categorical families and their derived combos.
def _select_target_agg_keys(df: pd.DataFrame) -> list[str]:
    preferred = list(BASE_TARGET_AGG_KEYS)
    pair_cols = sorted(
        col for col in df.columns
        if col.endswith("_Pair")
    )
    combo_cols = sorted(
        col for col in df.columns
        if col.endswith("_Combo")
    )
    bucket_cols = [
        col for col in ["Soil_pH_Band", "Moisture_Band", "Rainfall_Band", "Temperature_Band"]
        if col in df.columns
    ]

    seen = set()
    selected = []
    for col in preferred + combo_cols + pair_cols + bucket_cols:
        if col in df.columns and col not in seen:
            selected.append(col)
            seen.add(col)

    return selected


# Build leakage-safe OOF target encodings for train and full-train encodings for test.
def _add_target_encoding_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
    target_agg_keys: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = train_features.copy()
    test_encoded = test_features.copy()

    y = y.reset_index(drop=True).astype(int)
    global_target_mean = float(y.mean())
    global_target_std = float(y.std()) if len(y) > 1 else 0.0
    global_class_rates = {
        cls: float((y == cls).mean()) for cls in sorted(y.unique())
    }

    effective_splits = _resolve_stratified_splits(y, n_splits)
    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=seed,
    )

    for group_col in target_agg_keys:
        te_mean = pd.Series(np.nan, index=train_encoded.index)
        te_std = pd.Series(np.nan, index=train_encoded.index)
        te_count = pd.Series(np.nan, index=train_encoded.index)
        te_class_rates = {
            cls: pd.Series(np.nan, index=train_encoded.index)
            for cls in sorted(global_class_rates)
        }

        for fit_idx, valid_idx in splitter.split(train_features, y):
            fit_frame = pd.DataFrame(
                {
                    group_col: train_features.iloc[fit_idx][group_col].values,
                    "_target_num": y.iloc[fit_idx].values,
                }
            )

            agg = fit_frame.groupby(group_col)["_target_num"].agg(["mean", "std", "count"])
            valid_keys = train_features.iloc[valid_idx][group_col]
            te_mean.iloc[valid_idx] = valid_keys.map(agg["mean"])
            te_std.iloc[valid_idx] = valid_keys.map(agg["std"])
            te_count.iloc[valid_idx] = valid_keys.map(agg["count"])

            for cls in sorted(global_class_rates):
                class_rate = fit_frame.assign(
                    _class_flag=(fit_frame["_target_num"] == cls).astype(float)
                ).groupby(group_col)["_class_flag"].mean()
                te_class_rates[cls].iloc[valid_idx] = valid_keys.map(class_rate)

        train_encoded[f"TE_{group_col}_target_mean"] = te_mean.fillna(global_target_mean)
        train_encoded[f"TE_{group_col}_target_std"] = te_std.fillna(global_target_std)
        train_encoded[f"TE_{group_col}_target_count"] = te_count.fillna(1.0)
        for cls, fallback in global_class_rates.items():
            train_encoded[f"TE_{group_col}_class_{cls}_rate"] = (
                te_class_rates[cls].fillna(fallback)
            )

        full_frame = pd.DataFrame(
            {
                group_col: train_features[group_col].values,
                "_target_num": y.values,
            }
        )
        # Test rows can safely use statistics from the full training set.
        full_agg = full_frame.groupby(group_col)["_target_num"].agg(["mean", "std", "count"])
        test_encoded[f"TE_{group_col}_target_mean"] = (
            test_features[group_col].map(full_agg["mean"]).fillna(global_target_mean)
        )
        test_encoded[f"TE_{group_col}_target_std"] = (
            test_features[group_col].map(full_agg["std"]).fillna(global_target_std)
        )
        test_encoded[f"TE_{group_col}_target_count"] = (
            test_features[group_col].map(full_agg["count"]).fillna(1.0)
        )

        for cls, fallback in global_class_rates.items():
            full_class_rate = full_frame.assign(
                _class_flag=(full_frame["_target_num"] == cls).astype(float)
            ).groupby(group_col)["_class_flag"].mean()
            test_encoded[f"TE_{group_col}_class_{cls}_rate"] = (
                test_features[group_col].map(full_class_rate).fillna(fallback)
            )

    return train_encoded, test_encoded


# Main public entry point: build shared train/test features first, then append target-aware signals safely.
def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train = train_df.copy()
    test = test_df.copy()

    combined = pd.concat(
        [
            train.drop(columns=[TARGET], errors="ignore"),
            test,
        ],
        axis=0,
        ignore_index=True,
    )

    # Order matters here: stage/season ordinals are created before domain water-stress features use them.
    _add_bucket_features(combined)
    _add_ordinal_features(combined)
    _add_interaction_features(combined)
    _add_irrigation_domain_features(combined)
    _add_cross_features(combined)
    _add_pairwise_categorical_features(combined)
    _add_missingness_features(combined)
    _add_rounding_features(combined)
    _add_digit_features(combined)
    _add_anchor_combo_features(combined)
    combined = _apply_groupby_numeric_stats(combined)
    combined = _apply_groupby_quantiles(combined)
    combined = _add_groupby_ratio_features(combined)

    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    _fill_combined_frame(combined)

    train_features = combined.iloc[: len(train)].copy()
    test_features = combined.iloc[len(train):].copy()

    if y is not None:
        target_agg_keys = _select_target_agg_keys(train_features)
        train_features, test_features = _add_target_encoding_features(
            train_features=train_features,
            test_features=test_features,
            y=y,
            n_splits=n_splits,
            seed=seed,
            target_agg_keys=target_agg_keys,
        )

    train_features, test_features, final_categorical_cols, final_numerical_cols = (
        _finalize_train_test_frames(train_features, test_features)
    )

    return train_features, test_features, final_categorical_cols, final_numerical_cols


# ── Inference support ──────────────────────────────────────────────────────────

def _compute_groupby_numeric_lookups(df: pd.DataFrame) -> dict[str, "pd.DataFrame"]:
    lookups: dict[str, pd.DataFrame] = {}
    for group_col, config in NUMERIC_AGG_PLAN.items():
        if group_col not in df.columns:
            continue
        cols_present = [c for c in config["cols"] if c in df.columns]
        if not cols_present:
            continue
        grouped = df.groupby(group_col, dropna=False)[cols_present].agg(config["stats"])
        grouped.columns = [
            f"FE_{group_col}_{value_col}_{stat}"
            for value_col, stat in grouped.columns
        ]
        lookups[group_col] = grouped.reset_index()
    return lookups


def _compute_groupby_quantile_lookups(df: pd.DataFrame) -> dict[tuple, "pd.DataFrame"]:
    lookups: dict[tuple, pd.DataFrame] = {}
    for group_col, value_cols in GROUP_QUANTILE_PLAN.items():
        if group_col not in df.columns:
            continue
        for value_col in value_cols:
            if value_col not in df.columns:
                continue
            quantile_frame = (
                df.groupby(group_col, dropna=False)[value_col]
                .quantile([q / 100.0 for q in GROUP_QUANTILES])
                .unstack()
            )
            quantile_frame.columns = [
                f"FE_{group_col}_{value_col}_q{quantile}"
                for quantile in GROUP_QUANTILES
            ]
            lookups[(group_col, value_col)] = quantile_frame.reset_index()
    return lookups


def _compute_factorize_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for cat_col in BASE_CATEGORICAL_COLS:
        if cat_col not in df.columns:
            continue
        _, uniques = pd.factorize(_categorical_key(df[cat_col]), sort=True)
        maps[cat_col] = {str(v): int(i) for i, v in enumerate(uniques)}
    return maps


def _compute_te_lookups(
    df: pd.DataFrame, y: pd.Series, target_agg_keys: list[str]
) -> tuple[dict, float, float, dict]:
    y = y.reset_index(drop=True).astype(int)
    global_target_mean = float(y.mean())
    global_target_std = float(y.std()) if len(y) > 1 else 0.0
    global_class_rates = {int(cls): float((y == cls).mean()) for cls in sorted(y.unique())}

    lookups: dict = {}
    for group_col in target_agg_keys:
        if group_col not in df.columns:
            continue
        full_frame = pd.DataFrame({
            group_col: df[group_col].values,
            "_target_num": y.values,
        })
        full_agg = full_frame.groupby(group_col)["_target_num"].agg(["mean", "std", "count"])
        entry: dict = {
            "target_mean": full_agg["mean"],
            "target_std": full_agg["std"],
            "target_count": full_agg["count"],
        }
        for cls in sorted(global_class_rates):
            class_rate = (
                full_frame
                .assign(_class_flag=(full_frame["_target_num"] == cls).astype(float))
                .groupby(group_col)["_class_flag"].mean()
            )
            entry[f"class_{cls}_rate"] = class_rate
        lookups[group_col] = entry

    return lookups, global_target_mean, global_target_std, global_class_rates


def build_feature_artifacts(X_train: pd.DataFrame, y: pd.Series) -> dict:
    """Compute all lookup tables needed to reproduce feature engineering at inference time.

    Call this after build_features() returns X_train, then pickle the result to disk.
    """
    target_agg_keys = _select_target_agg_keys(X_train)
    te_lookups, global_target_mean, global_target_std, global_class_rates = _compute_te_lookups(
        X_train, y, target_agg_keys
    )
    num_cols = _resolve_numeric_columns(X_train)

    return {
        "groupby_numeric_lookups": _compute_groupby_numeric_lookups(X_train),
        "groupby_quantile_lookups": _compute_groupby_quantile_lookups(X_train),
        "factorize_maps": _compute_factorize_maps(X_train),
        "te_lookups": te_lookups,
        "global_te_stats": {
            "target_mean": global_target_mean,
            "target_std": global_target_std,
            "class_rates": global_class_rates,
        },
        "train_numeric_medians": {col: float(X_train[col].median()) for col in num_cols},
        "cat_cols": _resolve_categorical_columns(X_train),
        "num_cols": num_cols,
        "feature_cols": X_train.columns.tolist(),
    }


def _add_anchor_combo_features_inference(df: pd.DataFrame, factorize_maps: dict) -> None:
    """Anchor combo features using saved factorize mappings instead of pd.factorize."""
    for cat_col in BASE_CATEGORICAL_COLS:
        if cat_col not in df.columns or cat_col not in factorize_maps:
            continue
        val_map = factorize_maps[cat_col]
        factorized = (
            _categorical_key(df[cat_col])
            .map(val_map)
            .fillna(-1)
            .astype(np.int32)
        )
        factorized = pd.Series(factorized.values, index=df.index, dtype=np.int32)
        df[f"{cat_col}_factorized"] = factorized
        for anchor_col in ANCHOR_NUMERIC_COLS:
            if anchor_col in df.columns:
                df[f"{cat_col}_{anchor_col}_combo"] = (
                    factorized.astype(np.float64) * 1000.0 + df[anchor_col]
                )


def build_single_inference_features(row: dict, artifacts: dict) -> "pd.DataFrame":
    """Transform a raw input dict into a model-ready single-row DataFrame.

    The ``artifacts`` dict must be the one produced by ``build_feature_artifacts``
    and loaded from disk at serving startup.
    """
    df = pd.DataFrame([row])

    # Pydantic gives None for missing optional fields; pandas keeps those as object dtype.
    # Coerce known numeric columns to float so NaN-safe arithmetic works throughout.
    for col in BASE_NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _add_bucket_features(df)
    _add_ordinal_features(df)
    _add_interaction_features(df)
    _add_irrigation_domain_features(df)
    _add_cross_features(df)
    _add_pairwise_categorical_features(df)
    _add_missingness_features(df)
    _add_rounding_features(df)
    _add_digit_features(df)
    _add_anchor_combo_features_inference(df, artifacts["factorize_maps"])

    for group_col, lookup_df in artifacts["groupby_numeric_lookups"].items():
        if group_col in df.columns:
            df = df.merge(lookup_df, on=group_col, how="left")

    for (group_col, _val_col), lookup_df in artifacts["groupby_quantile_lookups"].items():
        if group_col in df.columns:
            df = df.merge(lookup_df, on=group_col, how="left")

    df = _add_groupby_ratio_features(df)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    global_stats = artifacts["global_te_stats"]
    for group_col, te_entry in artifacts["te_lookups"].items():
        if group_col not in df.columns:
            continue
        keys = df[group_col]
        df[f"TE_{group_col}_target_mean"] = (
            keys.map(te_entry["target_mean"]).fillna(global_stats["target_mean"])
        )
        df[f"TE_{group_col}_target_std"] = (
            keys.map(te_entry["target_std"]).fillna(global_stats["target_std"])
        )
        df[f"TE_{group_col}_target_count"] = (
            keys.map(te_entry["target_count"]).fillna(1.0)
        )
        for cls, fallback in global_stats["class_rates"].items():
            df[f"TE_{group_col}_class_{cls}_rate"] = (
                keys.map(te_entry[f"class_{cls}_rate"]).fillna(fallback)
            )

    medians = artifacts["train_numeric_medians"]
    for col in artifacts["num_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0.0))
    for col in artifacts["cat_cols"]:
        if col in df.columns:
            df[col] = _categorical_key(df[col])

    feature_cols = artifacts["feature_cols"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[feature_cols]
