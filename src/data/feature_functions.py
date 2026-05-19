try:
    import cudf as pd
    import cupy as np
    GPU = True
    print("cuDF/cuPy loaded — running on GPU")
except ImportError:
    import numpy as np
    import pandas as pd
    GPU = False
    print("cuDF not available — running on CPU")


# ── Raw categorical columns ──────────────────────────────────────────────────
CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

PAIRS = [
    ("Crop_Type",         "Season"),
    ("Crop_Type",         "Crop_Growth_Stage"),
    ("Region",            "Season"),
    ("Soil_Type",         "Region"),
    ("Crop_Type",         "Soil_Type"),
    ("Season",            "Soil_Type"),
    ("Irrigation_Type",   "Crop_Type"),
    ("Water_Source",      "Region"),
    ("Mulching_Used",     "Soil_Type"),
    ("Crop_Growth_Stage", "Season"),
]

TRIPLETS = [
    ("Crop_Type", "Season",            "Region"),
    ("Crop_Type", "Crop_Growth_Stage", "Season"),
    ("Soil_Type", "Region",            "Season"),
    ("Crop_Type", "Soil_Type",         "Region"),
]

NUM_COLS = [
    "Temperature_C", "Rainfall_mm", "Wind_Speed_kmh", "Humidity",
    "Sunlight_Hours", "Soil_Moisture", "Soil_pH", "Organic_Carbon",
    "Electrical_Conductivity", "Previous_Irrigation_mm", "Field_Area_hectare"
]


def add_formula_features(train, test):
    for df in [train, test]:
        high_score = (
            2 * (df['Soil_Moisture'] < 25).astype(int) +
            2 * (df['Rainfall_mm'] < 300).astype(int) +
            1 * (df['Temperature_C'] > 30).astype(int) +
            1 * (df['Wind_Speed_kmh'] > 10).astype(int)
        )
        low_score = (
            2 * (df['Crop_Growth_Stage'] == 'Harvest').astype(int) +
            2 * (df['Crop_Growth_Stage'] == 'Sowing').astype(int) +
            1 * (df['Mulching_Used'] == 'Yes').astype(int)
        )
        df['formula_score']        = high_score - low_score
        df['formula_pred']         = np.where(df['formula_score'] <= 0, 0,
                                     np.where(df['formula_score'] <= 3, 1, 2))
        df['dist_to_low_boundary'] = np.abs(df['formula_score'])
    return train, test


def add_binning_features(df):
    bins = {
        "Soil_pH"                 : [4.8, 5, 5.5, 6, 6.5, 7, 7.5, 8.5],
        "Soil_Moisture"           : [8, 12, 20, 25, 30, 32, 40, 50, 60, 70],
        "Organic_Carbon"          : [0.3, 0.4, 0.6, 0.9, 1, 1.3, 1.4, 1.6],
        "Electrical_Conductivity" : [0.1, 1, 1.7, 2.5, 3.5],
        "Temperature_C"           : [12, 20, 27, 35, 42],
        "Humidity"                : [25, 30, 40, 50, 62, 82, 95],
        "Rainfall_mm"             : [0, 400, 800, 1000, 1500, 2000, 2150, 2500],
        "Sunlight_Hours"          : [4, 5, 6, 7, 9, 10, 11],
        "Wind_Speed_kmh"          : [0.5, 5, 10, 20],
        "Field_Area_hectare"      : [0.3, 2.5, 5, 7.5, 10, 12.5, 15],
        "Previous_Irrigation_mm"  : [0.02, 20, 40, 62, 120],
    }
    for col, edges in bins.items():
        df[f"{col}_bin"] = pd.cut(
            df[col], bins=edges, labels=False, include_lowest=True
        ).astype("float32")
    return df


def add_domain_features(df):
    df = df.copy()
    T  = df["Temperature_C"]
    RH = df["Humidity"]
    R  = df["Rainfall_mm"]
    W  = df["Wind_Speed_kmh"]
    SH = df["Sunlight_Hours"]
    SM = df["Soil_Moisture"]
    PI = df["Previous_Irrigation_mm"]

    e_s = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    e_a = (RH / 100.0) * e_s
    df["vpd"]            = e_s - e_a
    df["vpd_normalized"] = df["vpd"] / (e_s + 1e-6)

    Ra = SH * 0.0820
    df["et0"]         = 0.0023 * (T + 17.8) * Ra
    df["et0_monthly"] = df["et0"] * 30

    KC_MAP = {"Sowing": 0.4, "Vegetative": 0.8, "Flowering": 1.15, "Harvest": 0.6}
    kc = df["Crop_Growth_Stage"].map(KC_MAP).fillna(0.8)
    df["crop_water_demand"]    = kc * df["et0"]
    df["crop_water_demand_mo"] = kc * df["et0_monthly"]

    df["effective_rainfall"]  = np.where(
        R <= 75, R * (125 - 0.2 * R) / 125, 125 + 0.1 * R
    )
    df["net_irrigation_need"] = df["crop_water_demand_mo"] - df["effective_rainfall"]
    df["irrigation_deficit"]  = np.clip(df["net_irrigation_need"], 0, None)
    df["water_balance"]       = R + PI - df["et0_monthly"]
    df["water_balance_sign"]  = np.sign(df["water_balance"])
    df["soil_water_deficit"]     = 25 - SM
    df["soil_water_deficit_pct"] = df["soil_water_deficit"] / 25.0
    df["aridity_index"]          = R / (T + 10 + 1e-6)
    df["dew_point"]              = T - ((100 - RH) / 5.0)
    df["temp_dew_spread"]        = T - df["dew_point"]
    df["wind_evap_factor"]       = df["et0"] * (1 + 0.1 * W)
    df["heat_wind_stress"]       = (T / 30.0) * (W / 10.0) * (1 - RH / 100.0)
    return df


def add_category_crosses(train, test):
    train = train.copy()
    test  = test.copy()
    new_cols = []

    def make_cross(df, cols):
        return df[cols[0]].astype(str).str.cat([df[c].astype(str) for c in cols[1:]], sep="__")

    for (a, b) in PAIRS:
        col_name = f"{a}__{b}"
        train[col_name] = make_cross(train, [a, b])
        test[col_name]  = make_cross(test,  [a, b])
        new_cols.append(col_name)

    for (a, b, c) in TRIPLETS:
        col_name = f"{a}__{b}__{c}"
        train[col_name] = make_cross(train, [a, b, c])
        test[col_name]  = make_cross(test,  [a, b, c])
        new_cols.append(col_name)

    all_cat_cols = CAT_COLS + new_cols.copy()

    print("Count encoding...")
    for col in all_cat_cols:
        freq    = train[col].value_counts()
        ce_name = f"ce_{col}"
        train[ce_name] = train[col].map(freq).fillna(0).astype(np.float32)
        test[ce_name]  = test[col].map(freq).fillna(0).astype(np.float32)
        new_cols.append(ce_name)

    print(f"Category crosses + count encoding done. New cols: {len(new_cols)}")
    return train, test, new_cols


def add_decimal_split(df):
    for col in NUM_COLS:
        df[f"{col}_int"] = np.floor(df[col])
        df[f"{col}_dec"] = df[col] - np.floor(df[col])
    return df


def add_digit_features(df, max_vals):
    for c in NUM_COLS:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype("int8")
        if max_vals[c] < 10:
            df[c] = df[c].round(3)
        elif max_vals[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    return df


def drop_constant_cols(train_df, test_df):
    drop = [c for c in test_df.columns if test_df[c].nunique() == 1]
    print(f"Dropping constant cols: {drop}")
    return train_df.drop(columns=drop, errors="ignore"), \
           test_df.drop(columns=drop, errors="ignore")


def frequency_encode(train_df, test_df, cols, min_freq=5):
    for c in cols:
        freq    = train_df[c].value_counts()
        mapping = {val: idx for idx, (val, _) in enumerate(freq[freq >= min_freq].items())}
        default = len(mapping)
        train_df[c] = train_df[c].map(lambda x, m=mapping, d=default: m.get(x, d))
        test_df[c]  = test_df[c].map(lambda x, m=mapping, d=default: m.get(x, d))
    return train_df, test_df


def add_interaction_features(df):
    T  = df["Temperature_C"]
    RH = df["Humidity"]
    R  = df["Rainfall_mm"]
    W  = df["Wind_Speed_kmh"]
    SM = df["Soil_Moisture"]
    PI = df["Previous_Irrigation_mm"]
    SH = df["Sunlight_Hours"]
    OC = df["Organic_Carbon"]
    EC = df["Electrical_Conductivity"]
    FA = df["Field_Area_hectare"]
    PH = df["Soil_pH"]

    # ── Ratios ───────────────────────────────────────────────────────────────
    df["T_RH_ratio"]     = T  / (RH + 1e-6)
    df["SM_R_ratio"]     = SM / (R  + 1e-6)
    df["PI_FA_ratio"]    = PI / (FA + 1e-6)
    df["W_RH_ratio"]     = W  / (RH + 1e-6)
    df["R_T_ratio"]      = R  / (T  + 1e-6)
    df["SM_PI_ratio"]    = SM / (PI + 1e-6)
    df["SH_RH_ratio"]    = SH / (RH + 1e-6)
    df["OC_EC_ratio"]    = OC / (EC + 1e-6)
    df["T_W_ratio"]      = T  / (W  + 1e-6)
    df["R_FA_ratio"]     = R  / (FA + 1e-6)
    df["SM_T_ratio"]     = SM / (T  + 1e-6)
    df["PI_R_ratio"]     = PI / (R  + 1e-6)
    df["SH_W_ratio"]     = SH / (W  + 1e-6)
    df["PH_EC_ratio"]    = PH / (EC + 1e-6)
    df["OC_SM_ratio"]    = OC / (SM + 1e-6)
    df["FA_PI_ratio"]    = FA / (PI + 1e-6)
    df["RH_W_ratio"]     = RH / (W  + 1e-6)
    df["T_SH_ratio"]     = T  / (SH + 1e-6)

    # ── Products ─────────────────────────────────────────────────────────────
    df["T_W_product"]    = T  * W
    df["T_SH_product"]   = T  * SH
    df["W_SH_product"]   = W  * SH
    df["SM_OC_product"]  = SM * OC
    df["R_RH_product"]   = R  * RH
    df["T_RH_product"]   = T  * RH
    df["PI_FA_product"]  = PI * FA
    df["SM_R_product"]   = SM * R
    df["EC_PH_product"]  = EC * PH
    df["W_T_RH_product"] = W  * T * (1 - RH / 100)
    df["SM_T_product"]   = SM * T
    df["OC_PH_product"]  = OC * PH
    df["SH_T_product"]   = SH * T
    df["W_RH_product"]   = W  * RH
    df["R_SH_product"]   = R  * SH
    df["PI_SM_product"]  = PI * SM
    df["EC_SM_product"]  = EC * SM
    df["FA_W_product"]   = FA * W

    # ── Differences ──────────────────────────────────────────────────────────
    df["SM_minus_25"]    = SM - 25
    df["R_minus_300"]    = R  - 300
    df["PI_minus_SM"]    = PI - SM
    df["T_minus_27"]     = T  - 27
    df["W_minus_10"]     = W  - 10
    df["R_minus_PI"]     = R  - PI
    df["SM_minus_OC"]    = SM - OC
    df["T_minus_RH"]     = T  - RH
    df["SH_minus_W"]     = SH - W
    df["PH_minus_7"]     = PH - 7.0

    # ── Powers ───────────────────────────────────────────────────────────────
    df["SM_sq"]          = SM ** 2
    df["T_sq"]           = T  ** 2
    df["R_sq"]           = R  ** 2
    df["W_sq"]           = W  ** 2
    df["RH_sq"]          = RH ** 2
    df["SH_sq"]          = SH ** 2
    df["SM_cb"]          = SM ** 3
    df["T_cb"]           = T  ** 3
    df["SM_sqrt"]        = np.sqrt(np.clip(SM, 0, None))
    df["R_sqrt"]         = np.sqrt(np.clip(R,  0, None))
    df["T_sqrt"]         = np.sqrt(np.clip(T,  0, None))
    df["W_sqrt"]         = np.sqrt(np.clip(W,  0, None))

    # ── Logarithms ───────────────────────────────────────────────────────────
    df["log_R"]          = np.log1p(R)
    df["log_PI"]         = np.log1p(PI)
    df["log_FA"]         = np.log1p(FA)
    df["log_SM"]         = np.log1p(SM)
    df["log_W"]          = np.log1p(W)
    df["log_T"]          = np.log1p(T)
    df["log_SH"]         = np.log1p(SH)
    df["log_EC"]         = np.log1p(EC)
    df["log_OC"]         = np.log1p(OC)

    # ── Exponentials (scaled) ─────────────────────────────────────────────────
    df["exp_SM_scaled"]  = np.exp(np.clip(SM / 100, -10, 10))
    df["exp_T_scaled"]   = np.exp(np.clip(T  / 100, -10, 10))
    df["exp_RH_scaled"]  = np.exp(np.clip(RH / 100, -10, 10))

    # ── Trigonometric ────────────────────────────────────────────────────────
    df["sin_T"]          = np.sin(T  * np.pi / 180)
    df["cos_T"]          = np.cos(T  * np.pi / 180)
    df["sin_RH"]         = np.sin(RH * np.pi / 180)
    df["cos_RH"]         = np.cos(RH * np.pi / 180)
    df["sin_SM"]         = np.sin(SM * np.pi / 180)
    df["cos_SM"]         = np.cos(SM * np.pi / 180)
    df["sin_W"]          = np.sin(W  * np.pi / 180)
    df["cos_W"]          = np.cos(W  * np.pi / 180)

    # ── Threshold distances ───────────────────────────────────────────────────
    df["SM_dist_25"]     = np.abs(SM - 25)
    df["R_dist_300"]     = np.abs(R  - 300)
    df["T_dist_30"]      = np.abs(T  - 30)
    df["W_dist_10"]      = np.abs(W  - 10)
    df["PH_dist_7"]      = np.abs(PH - 7.0)
    df["SM_dist_25_sq"]  = (SM - 25)  ** 2
    df["R_dist_300_sq"]  = (R  - 300) ** 2
    df["T_dist_30_sq"]   = (T  - 30)  ** 2
    df["W_dist_10_sq"]   = (W  - 10)  ** 2

    # ── Sigmoid transforms ────────────────────────────────────────────────────
    df["sigmoid_SM_25"]  = 1 / (1 + np.exp(-(SM - 25)))
    df["sigmoid_R_300"]  = 1 / (1 + np.exp(-(R  - 300) / 50))
    df["sigmoid_T_30"]   = 1 / (1 + np.exp(-(T  - 30)))
    df["sigmoid_W_10"]   = 1 / (1 + np.exp(-(W  - 10)))

    # ── Harmonic means ────────────────────────────────────────────────────────
    df["harm_SM_R"]      = 2 * SM * R  / (SM + R  + 1e-6)
    df["harm_T_RH"]      = 2 * T  * RH / (T  + RH + 1e-6)
    df["harm_W_SH"]      = 2 * W  * SH / (W  + SH + 1e-6)

    # ── Geometric means ───────────────────────────────────────────────────────
    df["geom_SM_R"]      = np.sqrt(np.clip(SM * R,  0, None))
    df["geom_T_SH"]      = np.sqrt(np.clip(T  * SH, 0, None))
    df["geom_W_RH"]      = np.sqrt(np.clip(W  * RH, 0, None))

    # ── Min / Max ─────────────────────────────────────────────────────────────
    df["min_SM_PI"]      = np.minimum(SM, PI)
    df["max_SM_PI"]      = np.maximum(SM, PI)
    df["min_T_RH"]       = np.minimum(T,  RH)
    df["max_T_RH"]       = np.maximum(T,  RH)
    df["min_R_W"]        = np.minimum(R,  W)
    df["max_R_W"]        = np.maximum(R,  W)

    # ── Triple interactions ───────────────────────────────────────────────────
    df["SM_R_T"]         = SM * R  * T
    df["W_T_SH"]         = W  * T  * SH
    df["SM_OC_PH"]       = SM * OC * PH
    df["R_RH_W"]         = R  * RH * W
    df["T_W_RH"]         = T  * W  * RH
    df["SM_T_W"]         = SM * T  * W
    df["R_T_SH"]         = R  * T  * SH
    df["SM_R_RH"]        = SM * R  * RH

    return df


def run_all_features(train, test):
    print("1. Formula features...")
    train, test = add_formula_features(train, test)

    print("2. Binning features...")
    train = add_binning_features(train)
    test  = add_binning_features(test)

    print("3. Decimal split...")
    train = add_decimal_split(train)
    test  = add_decimal_split(test)

    print("4. Digit features...")
    max_vals = train[NUM_COLS].max().to_dict()
    train = add_digit_features(train, max_vals)
    test  = add_digit_features(test,  max_vals)

    print("5. Domain features...")
    train = add_domain_features(train)
    test  = add_domain_features(test)

    print("6. Interaction features...")
    train = add_interaction_features(train)
    test  = add_interaction_features(test)

    print("7. Category crosses + count encoding...")
    train, test, _ = add_category_crosses(train, test)

    print("8. Drop constant cols...")
    train, test = drop_constant_cols(train, test)

    print(f"\nDone. Train: {train.shape} | Test: {test.shape}")
    return train, test
