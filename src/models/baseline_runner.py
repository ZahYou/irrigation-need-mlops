import json
import os
import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from catboost import CatBoostClassifier
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
with open("configs/train_config.yaml") as _f:
    _cfg = yaml.safe_load(_f)

SEED        = _cfg["training"]["seed"]
N_SPLITS    = _cfg["training"]["n_splits"]
_ENABLED    = _cfg["training"]["models"]
N_CLASSES   = 3
VERBOSE     = 100
TARGET      = "Irrigation_Need"
ID_COL      = "id"
CLASS_NAMES = ["Low", "Medium", "High"]

os.makedirs("saved_oof",          exist_ok=True)
os.makedirs("saved_models",       exist_ok=True)
os.makedirs("submissions",        exist_ok=True)
os.makedirs("feature_importance", exist_ok=True)

_mlflow_cfg = _cfg.get("mlflow", {})
mlflow.set_tracking_uri(_mlflow_cfg.get("tracking_uri", "mlruns"))
mlflow.set_experiment(_mlflow_cfg.get("experiment_name", "irrigation-need"))

_MODEL_PARAM_SECTION = {
    "xgb":      "xgboost",
    "lgbm":     "lightgbm",
    "catboost": "catboost",
    "logreg":   None,
}

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Load pre-built features from the DVC features stage.
X_train  = pd.read_pickle("data/processed/X_train.pkl")
X_test   = pd.read_pickle("data/processed/X_test.pkl")
y_train  = pd.read_pickle("data/processed/y_train.pkl")

with open("data/processed/cat_cols.json") as _f:
    cat_cols = json.load(_f)
with open("data/processed/num_cols.json") as _f:
    num_cols = json.load(_f)

# Use IDs from the processed X_test so they match the (possibly dev-sampled) predictions.
test = X_test[[ID_COL]].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
# Label-encoded version for XGBoost / LogReg
le_dict = {}
X_train_le = X_train.copy()
X_test_le  = X_test.copy()
for col in cat_cols:
    le = LabelEncoder()
    combined_values = pd.concat(
        [X_train[col].astype(str), X_test[col].astype(str)],
        axis=0,
        ignore_index=True,
    )
    le.fit(combined_values)
    X_train_le[col] = le.transform(X_train[col].astype(str))
    X_test_le[col]  = le.transform(X_test[col].astype(str))
    le_dict[col]    = le

with open("saved_models/label_encoders.pkl", "wb") as _le_f:
    pickle.dump(le_dict, _le_f)

# Category dtype version for LightGBM / CatBoost
X_train_cat = X_train.copy()
X_test_cat  = X_test.copy()
for col in cat_cols:
    X_train_cat[col] = X_train_cat[col].astype("category")
    X_test_cat[col]  = X_test_cat[col].astype("category")

# ─────────────────────────────────────────────────────────────
# run_cv
# ─────────────────────────────────────────────────────────────
def run_cv(model_fn, name, X_tr_full, X_te_full, use_cat=False):
    proba_path = f"saved_oof/{name}_proba.npy"
    class_path = f"saved_oof/{name}_class.npy"
    test_path  = f"saved_oof/{name}_test.npy"
    if os.path.exists(proba_path):
        print(f"  [skip] {name} already done — loading")
        oof_proba  = np.load(proba_path)
        oof_class  = np.load(class_path)
        test_proba = np.load(test_path)
        score = balanced_accuracy_score(y_train, oof_class)
        print(f"  [done] {name} — BA: {score:.6f}")
        return {"oof_proba": oof_proba, "oof_class": oof_class,
                "test_proba": test_proba, "score": score}

    oof_proba  = np.zeros((len(X_tr_full), N_CLASSES), dtype=np.float64)
    test_preds = []
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_full, y_train)):
        print(f"  Fold {fold+1}/{N_SPLITS}...", end=" ")

        X_tr  = X_tr_full.iloc[tr_idx]
        X_val = X_tr_full.iloc[val_idx]
        X_te  = X_te_full
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        model = model_fn()

        if use_cat and isinstance(model, CatBoostClassifier):
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val),
                      cat_features=cat_cols,
                      verbose=VERBOSE)
        elif isinstance(model, xgb.XGBClassifier):
            sw = compute_sample_weight("balanced", y_tr)
            model.fit(X_tr, y_tr,
                      sample_weight=sw,
                      eval_set=[(X_val, y_val)],
                      verbose=VERBOSE)
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)])

        val_proba = model.predict_proba(X_val)
        te_proba  = model.predict_proba(X_te)

        oof_proba[val_idx] = val_proba
        test_preds.append(te_proba)
        fold_models.append(model)

        # Log model to MLflow with framework-aware flavor + signature
        signature = infer_signature(X_val.head(5), val_proba[:5])
        if isinstance(model, CatBoostClassifier):
            mlflow.catboost.log_model(model, name=f"model_fold{fold+1}", signature=signature)
        elif isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, name=f"model_fold{fold+1}", signature=signature)
        elif isinstance(model, lgb.LGBMClassifier):
            mlflow.lightgbm.log_model(model, name=f"model_fold{fold+1}", signature=signature)
        else:
            mlflow.sklearn.log_model(model, name=f"model_fold{fold+1}", signature=signature)
        fold_score = balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1))

        print(f"Done — BA: {fold_score:.6f}")
        mlflow.log_metric("fold_ba", fold_score, step=fold)

        with open(f"saved_models/{name}_fold{fold+1}.pkl", "wb") as f:
            pickle.dump(model, f)

    test_proba = np.mean(test_preds, axis=0)
    oof_class  = np.argmax(oof_proba, axis=1)
    score      = balanced_accuracy_score(y_train, oof_class)

    print(f"  [done] {name} — BA: {score:.6f}")

    np.save(proba_path, oof_proba)
    np.save(class_path, oof_class)
    np.save(test_path,  test_proba)

    return {"oof_proba": oof_proba, "oof_class": oof_class,
            "test_proba": test_proba, "score": score,
            "models": fold_models}

# ─────────────────────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────────────────────
def make_submission(result, name):
    test_class = np.argmax(result["test_proba"], axis=1)
    inv_label_map  = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({
        ID_COL: test[ID_COL].values,
        TARGET: [inv_label_map[c] for c in test_class]
    })
    path = f"submissions/sub_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  [saved] Submission saved: {path}")
    print(sub[TARGET].value_counts())
    return sub

# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(result, name, feature_names, top_n=20):
    models = result.get("models", [])
    if not models:
        print(f"  [warn] No models in result for {name} — skipping FI")
        return

    model = models[0]

    if isinstance(model, xgb.XGBClassifier):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        title = f"XGBoost Feature Importance — {name}"

    elif isinstance(model, lgb.LGBMClassifier):
        fi = pd.Series(model.feature_importances_, index=feature_names)
        title = f"LightGBM Feature Importance — {name}"

    elif isinstance(model, CatBoostClassifier):
        fi = pd.Series(model.get_feature_importance(),
                       index=model.feature_names_)
        title = f"CatBoost Feature Importance — {name}"

    elif isinstance(model, Pipeline) and isinstance(model.named_steps.get("clf"), LogisticRegression):
        # For LogReg pipeline: mean absolute coefficient across classes
        fi = pd.Series(
            np.abs(model.named_steps["clf"].coef_).mean(axis=0),
            index=feature_names
        )
        title = f"LogReg Mean |Coef| — {name}"

    else:
        print(f"  [warn] Unknown model type for {name} — skipping FI")
        return

    fi = fi.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = ["#3cb371" if i >= len(fi) - 5 else "#a8d5b5"
              for i in range(len(fi))]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(alpha=0.4, axis="x")
    plt.tight_layout()
    plt.savefig(f"feature_importance/fi_{name}.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print(f"  [saved] Feature importance: feature_importance/fi_{name}.png")

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────
def make_xgb():
    return xgb.XGBClassifier(
        objective             = "multi:softprob",
        num_class             = N_CLASSES,
        n_estimators          = 1000,
        learning_rate         = 0.05,
        max_depth             = 6,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 20,
        tree_method           = "hist",
        random_state          = SEED,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 50,
        verbosity             = 0,
        enable_categorical    = True,
    )

def make_lgbm():
    return lgb.LGBMClassifier(
        objective         = "multiclass",
        num_class         = N_CLASSES,
        n_estimators      = 1000,
        learning_rate     = 0.05,
        num_leaves        = 63,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        class_weight      = "balanced",
        random_state      = SEED,
        verbosity         = -1,
        callbacks         = [lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(VERBOSE)],
    )

def make_cat():
    return CatBoostClassifier(
        loss_function         = "MultiClass",
        classes_count         = N_CLASSES,
        iterations            = 1000,
        learning_rate         = 0.05,
        depth                 = 6,
        l2_leaf_reg           = 3,
        random_strength       = 1,
        bagging_temperature   = 1,
        auto_class_weights    = "Balanced",
        random_seed           = SEED,
        early_stopping_rounds = 50,
        verbose               = VERBOSE,
    )

def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class  = "multinomial",
            solver       = "lbfgs",
            max_iter     = 2000,
            C            = 1.0,
            class_weight = "balanced",
            random_state = SEED,
            n_jobs       = -1,
        )),
    ])

# ─────────────────────────────────────────────────────────────
# CONFIGS REGISTRY
# (name, model_fn, X_train, X_test, use_cat)
# ─────────────────────────────────────────────────────────────
_ALL_CONFIGS = {
    "xgb":      ("xgb_baseline_v001",    make_xgb,    X_train_le,  X_test_le,  False),
    "lgbm":     ("lgbm_baseline_v001",   make_lgbm,   X_train_cat, X_test_cat, True),
    "catboost": ("cat_baseline_v001",    make_cat,    X_train_cat, X_test_cat, True),
    "logreg":   ("logreg_baseline_v001", make_logreg, X_train_le,  X_test_le,  False),
}
CONFIGS = [_ALL_CONFIGS[m] for m in _ENABLED if m in _ALL_CONFIGS]

# ─────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────
all_models = {}

for model_key, (name, model_fn, X_tr, X_te, use_cat) in zip(_ENABLED, CONFIGS, strict=True):
    print(f"\n{'='*55}\n  {name}\n{'='*55}")

    with mlflow.start_run(run_name=name):
        # ── params ──────────────────────────────────────────
        mlflow.log_params({
            "model_type":  model_key,
            "seed":        SEED,
            "n_splits":    N_SPLITS,
            "n_features":  X_tr.shape[1],
            "train_size":  len(X_train),
            "test_size":   len(X_test),
            "dev_mode":    _cfg.get("dev_mode", False),
            "dev_sample":  _cfg["data"].get("dev_sample"),
        })
        param_section = _MODEL_PARAM_SECTION.get(model_key)
        if param_section and param_section in _cfg:
            mlflow.log_params(_cfg[param_section])

        # ── train / load ─────────────────────────────────────
        result = run_cv(model_fn, name, X_tr, X_te, use_cat)
        all_models[name] = result

        # ── metrics ──────────────────────────────────────────
        mlflow.log_metric("oof_ba", result["score"])

        # ── artifacts ────────────────────────────────────────
        make_submission(result, name)
        mlflow.log_artifact(f"submissions/sub_{name}.csv", artifact_path="submissions")

        plot_feature_importance(result, name, feature_names=X_tr.columns.tolist())
        fi_path = f"feature_importance/fi_{name}.png"
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  BASELINE SUMMARY")
print(f"{'='*55}")
for name, result in all_models.items():
    print(f"  {name:40s}  BA: {result['score']:.6f}")
