"""Train, calibrate and evaluate the Freddie Mac PD/LGD/ECL models.

The module intentionally consumes only the explicit modeling contract. Test
outcomes remain untouched until both model estimators and probability
calibrators have been fit on the development partitions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
import shutil
import warnings
from zipfile import ZipFile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SPLITS = ("train", "validation", "calibration", "test")
IDENTIFIER_COLUMNS = {
    "loan_identifier",
    "as_of_date",
    "source_cohort",
    "refinance_predecessor_id",
    "split",
}
LGD_FEATURES = ["current_ltv_proxy", "classic_fico", "original_dti",
                "original_interest_rate", "default_year", "property_state"]


@dataclass
class PlattCalibrator:
    """One-dimensional logistic recalibration on the raw PD logit."""

    coefficient: float
    intercept: float

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
        logit = np.log(probability / (1 - probability))
        return 1 / (1 + np.exp(-(self.intercept + self.coefficient * logit)))


def load_config(root: Path) -> dict:
    path = root / "config" / "modeling.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_contract(root: Path) -> dict:
    path = root / "data" / "modeling" / "modeling_contract.json"
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _target_metrics(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    fpr, tpr, _ = roc_curve(y, p)
    decile = pd.qcut(pd.Series(p), q=10, labels=False, duplicates="drop")
    grouped = pd.DataFrame({"p": p, "y": y, "decile": decile}).groupby("decile")
    ece = float(
        grouped.apply(
            lambda group: abs(group.p.mean() - group.y.mean()) * len(group),
            include_groups=False,
        ).sum()
        / len(y)
    )
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "actual_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "pred_over_actual": float(p.mean() / y.mean()) if y.mean() else None,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece_10": ece,
    }


def audit_data(root: Path, output_dir: Path) -> dict:
    """Validate key uniqueness, temporal order and forbidden sentinel values."""
    contract = load_contract(root)
    features = contract["feature_columns"]
    target = contract["target"]
    frames = {}
    loan_sets = {}
    predecessor_frames = []
    missing_by_split = {}
    rows = []
    for split in SPLITS:
        frame = pd.read_parquet(
            root / "data" / "modeling" / "pd" / f"{split}.parquet",
            columns=["loan_identifier", "as_of_date", "source_cohort",
                     "refinance_predecessor_id", target] + features,
        )
        if frame.loan_identifier.duplicated().any():
            raise ValueError(f"{split}: duplicate loan_identifier")
        if frame[target].isna().any():
            raise ValueError(f"{split}: missing target")
        missing = frame[features].isna().mean()
        missing_by_split[split] = {
            key: float(value) for key, value in missing[missing.gt(0)]
            .sort_values(ascending=False).items()
        }
        if not set(frame[target].unique()).issubset({0, 1}):
            raise ValueError(f"{split}: target is not binary")
        for column in frame.select_dtypes(include=np.number).columns:
            if frame[column].isin([999, 9999]).any():
                raise ValueError(f"{split}: unresolved sentinel in {column}")
        frames[split] = frame
        loan_sets[split] = set(frame.loan_identifier)
        predecessor_frames.append(
            frame.loc[frame.refinance_predecessor_id.notna(),
                      ["loan_identifier", "refinance_predecessor_id", "source_cohort"]]
            .assign(split=split)
        )
        rows.append({
            "split": split,
            "rows": len(frame),
            "positives": int(frame[target].sum()),
            "event_rate": float(frame[target].mean()),
            "first_anchor": frame.as_of_date.min(),
            "last_anchor": frame.as_of_date.max(),
            "cohorts": int(frame.source_cohort.nunique()),
        })

    overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = loan_sets[left] & loan_sets[right]
            overlaps[f"{left}__{right}"] = len(overlap)
    if any(overlaps.values()):
        raise ValueError(f"Loan overlap across partitions: {overlaps}")

    temporal = {
        "train_before_validation": frames["train"].as_of_date.max()
        < frames["validation"].as_of_date.min(),
        "development_before_test": max(
            frames[split].as_of_date.max()
            for split in ("train", "validation", "calibration")
        ) < frames["test"].as_of_date.min(),
    }
    if not all(temporal.values()):
        raise ValueError(f"Temporal partition order failed: {temporal}")

    predecessors = pd.concat(predecessor_frames, ignore_index=True)
    all_ids = set().union(*loan_sets.values())
    cross_split = predecessors[
        predecessors.refinance_predecessor_id.isin(all_ids)
    ].copy()
    if not cross_split.empty:
        split_lookup = {
            loan_id: split
            for split, ids in loan_sets.items()
            for loan_id in ids
        }
        cross_split["predecessor_split"] = cross_split.refinance_predecessor_id.map(split_lookup)
        leaked = cross_split[cross_split.predecessor_split.ne(cross_split.split)]
        if not leaked.empty:
            raise ValueError(f"Known cross-split refinance leakage: {len(leaked)}")

    report = {
        "contract": {
            "release": contract["release"],
            "target": target,
            "feature_count": len(features),
            "pd_ready_rows": contract["pd_ready_rows"],
            "performance_cutoff": contract["performance_cutoff"],
        },
        "partitions": rows,
        "pairwise_loan_overlap": overlaps,
        "temporal_checks": temporal,
        "feature_missingness": missing_by_split,
        "known_cross_split_refinance_leaks": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "pd_split_audit.csv", index=False)
    _write_json(output_dir / "audit.json", report)
    return report


def validate_source_sample(archive: Path, year: int = 2008) -> dict:
    """Check real nested sample widths and the signed-loss convention."""
    if not archive.is_file():
        return {"status": "skipped", "reason": f"archive not found: {archive}"}
    with ZipFile(archive) as outer:
        member = f"raw/sample_{year}.zip"
        if member not in outer.namelist():
            raise FileNotFoundError(f"{member} is absent from {archive}")
        nested = outer.read(member)
    with ZipFile(BytesIO(nested)) as sample:
        orig_member = f"sample_orig_{year}.txt"
        perf_member = f"sample_perf_{year}.txt"
        with sample.open(orig_member) as handle:
            origination = TextIOWrapper(handle, encoding="utf-8-sig")
            origination_rows = [
                next(origination).rstrip("\r\n").split("|") for _ in range(3)
            ]
        with sample.open(perf_member) as handle:
            performance = TextIOWrapper(handle, encoding="utf-8-sig")
            performance_rows = [
                next(performance).rstrip("\r\n").split("|") for _ in range(3)
            ]
            delinquency_codes = Counter()
            removal_codes = Counter()
            loss_signs = Counter()
            row_count = 0
            for line in performance:
                fields = line.rstrip("\r\n").split("|")
                if len(fields) != 35:
                    raise ValueError(
                        f"{perf_member}: expected 35 fields at row {row_count + 1}, "
                        f"got {len(fields)}"
                    )
                row_count += 1
                delinquency_codes[fields[3]] += 1
                if fields[8] in {"02", "03", "09"}:
                    removal_codes[fields[8]] += 1
                    if fields[21]:
                        value = float(fields[21])
                        loss_signs[
                            "positive" if value > 0 else "negative" if value < 0 else "zero"
                        ] += 1
    if any(len(row) != 31 for row in origination_rows):
        raise ValueError(f"{orig_member}: expected 31 fields")
    positive = loss_signs["positive"]
    resolved = sum(loss_signs.values())
    report = {
        "status": "pass",
        "archive": str(archive),
        "sample_year": year,
        "origination_width": 31,
        "performance_width": 35,
        "origination_first_values": origination_rows[0][:12],
        "performance_first_values": performance_rows[0][:14],
        "performance_rows_checked": row_count,
        "delinquency_codes": dict(delinquency_codes.most_common()),
        "zero_balance_removal_codes": dict(removal_codes),
        "actual_loss_signs_for_02_03_09": dict(loss_signs),
        "positive_loss_share": positive / resolved if resolved else None,
        "loss_sign_check": bool(resolved and positive / resolved >= 0.5),
    }
    if not report["loss_sign_check"]:
        raise ValueError(f"Loss sign convention failed: {report}")
    return report


def _feature_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    return numeric, categorical


def _logistic_pipeline(
    numeric: list[str], categorical: list[str], config: dict
) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                  keep_empty_features=True)),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=int(config["onehot_min_frequency"]),
            sparse_output=True,
            dtype=np.float32,
        )),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop", sparse_threshold=1.0)
    model = LogisticRegression(
        C=float(config["logistic"]["c"]),
        max_iter=int(config["logistic"]["max_iter"]),
        solver="lbfgs",
        random_state=int(config["seed"]),
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _category_levels(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    categorical: list[str],
) -> dict[str, pd.Index]:
    levels = {}
    for column in categorical:
        values = pd.concat([
            train[column].astype("string"),
            validation[column].astype("string"),
        ], ignore_index=True).fillna("__MISSING__")
        unique = pd.unique(values)
        if "__MISSING__" not in unique:
            unique = np.append(unique, "__MISSING__")
        if "__UNKNOWN__" not in unique:
            unique = np.append(unique, "__UNKNOWN__")
        levels[column] = pd.Index(unique)
    return levels


def _xgb_frame(
    frame: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    levels: dict[str, pd.Index],
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in numeric:
        output[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    for column in categorical:
        values = frame[column].astype("string")
        values = values.mask(values.isna() | values.eq(""), "__MISSING__")
        categories = levels[column]
        if "__MISSING__" not in categories:
            categories = categories.append(pd.Index(["__MISSING__"]))
        if "__UNKNOWN__" not in categories:
            categories = categories.append(pd.Index(["__UNKNOWN__"]))
        values = values.where(values.isin(categories), "__UNKNOWN__")
        output[column] = pd.Categorical(values, categories=categories)
    return output[features]


def _fit_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    config: dict,
):
    levels = _category_levels(train, validation, categorical)
    train_x = _xgb_frame(train, features, numeric, categorical, levels)
    validation_x = _xgb_frame(validation, features, numeric, categorical, levels)
    params = config["xgboost"]
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        max_bin=int(params["max_bin"]),
        early_stopping_rounds=int(params["early_stopping_rounds"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(config["seed"]),
        verbosity=0,
    )
    model.fit(
        train_x,
        train["serious_delinquency_12m"].astype(int),
        eval_set=[(
            validation_x,
            validation["serious_delinquency_12m"].astype(int),
        )],
        verbose=False,
    )
    return model, levels


def _predict_xgb(model, levels, frame, features, numeric, categorical) -> np.ndarray:
    matrix = _xgb_frame(frame, features, numeric, categorical, levels)
    return model.predict_proba(matrix)[:, 1]


def _fit_calibrator(
    raw_probability: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    holdout_fraction: float,
) -> tuple[object, str, pd.DataFrame]:
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    raw_probability = raw_probability[order]
    labels = labels[order]
    split_at = max(100, int(len(labels) * (1 - holdout_fraction)))
    split_at = min(split_at, len(labels) - 100)
    early_p, late_p = raw_probability[:split_at], raw_probability[split_at:]
    early_y, late_y = labels[:split_at], labels[split_at:]

    candidates = {}
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(early_p, early_y)
    candidates["isotonic"] = isotonic
    platt_fit = LogisticRegression(C=1e6, solver="lbfgs")
    early_logit = np.log(np.clip(early_p, 1e-8, 1 - 1e-8)
                         / (1 - np.clip(early_p, 1e-8, 1 - 1e-8)))
    platt_fit.fit(early_logit.reshape(-1, 1), early_y)
    candidates["platt"] = PlattCalibrator(
        coefficient=float(platt_fit.coef_[0, 0]),
        intercept=float(platt_fit.intercept_[0]),
    )

    selection = []
    for name, calibrator in candidates.items():
        predicted = np.asarray(calibrator.predict(late_p), dtype=float)
        selection.append({
            "method": name,
            "holdout_brier": float(brier_score_loss(late_y, predicted)),
            "holdout_log_loss": float(log_loss(late_y, predicted)),
            "holdout_rows": len(late_y),
        })
    selection_frame = pd.DataFrame(selection).sort_values("holdout_log_loss")
    chosen = selection_frame.iloc[0].method

    if chosen == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        calibrator.fit(raw_probability, labels)
    else:
        fit = LogisticRegression(C=1e6, solver="lbfgs")
        logit = np.log(np.clip(raw_probability, 1e-8, 1 - 1e-8)
                       / (1 - np.clip(raw_probability, 1e-8, 1 - 1e-8)))
        fit.fit(logit.reshape(-1, 1), labels)
        calibrator = PlattCalibrator(
            coefficient=float(fit.coef_[0, 0]),
            intercept=float(fit.intercept_[0]),
        )
    return calibrator, chosen, selection_frame


def _calibrated(calibrator, probability) -> np.ndarray:
    return np.clip(np.asarray(calibrator.predict(probability), dtype=float), 1e-8, 1 - 1e-8)


def train_pd(root: Path, config: dict, report_dir: Path, model_dir: Path) -> dict:
    contract = load_contract(root)
    features = contract["feature_columns"]
    target = contract["target"]
    frames = {
        split: pd.read_parquet(
            root / "data" / "modeling" / "pd" / f"{split}.parquet",
            columns=["loan_identifier", "as_of_date", "source_cohort",
                     "refinance_predecessor_id", target] + features,
        )
        for split in SPLITS
    }
    numeric, categorical = _feature_types(frames["train"], features)
    y = {split: frame[target].astype(int).to_numpy() for split, frame in frames.items()}

    print("Fitting logistic regression", flush=True)
    logistic = _logistic_pipeline(numeric, categorical, config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logistic.fit(frames["train"][features], y["train"])

    print("Fitting XGBoost", flush=True)
    xgboost, levels = _fit_xgboost(
        frames["train"], frames["validation"], features,
        numeric, categorical, config,
    )

    probabilities = {}
    for model_name in ("logistic", "xgboost"):
        probabilities[model_name] = {}
        for split, frame in frames.items():
            if model_name == "logistic":
                raw = logistic.predict_proba(frame[features])[:, 1]
            else:
                raw = _predict_xgb(
                    xgboost, levels, frame, features, numeric, categorical,
                )
            probabilities[model_name][f"{split}_raw"] = raw
        probabilities[model_name]["calibration_fit_raw"] = probabilities[model_name]["calibration_raw"]

    calibrators, calibration_selection = {}, []
    for model_name in ("logistic", "xgboost"):
        calibrator, method, selection = _fit_calibrator(
            probabilities[model_name]["calibration_fit_raw"],
            y["calibration"],
            frames["calibration"].as_of_date,
            float(config["calibration"]["holdout_fraction"]),
        )
        calibrators[model_name] = calibrator
        selection.insert(0, "model", model_name)
        calibration_selection.append(selection)
        for split in SPLITS:
            probabilities[model_name][f"{split}_calibrated"] = _calibrated(
                calibrator, probabilities[model_name][f"{split}_raw"],
            )
        print(f"{model_name} calibrator: {method}", flush=True)

    calibration_selection = pd.concat(calibration_selection, ignore_index=True)
    calibration_selection.to_csv(report_dir / "calibration_method_selection.csv", index=False)

    metric_rows = []
    for model_name in ("logistic", "xgboost"):
        for kind in ("raw", "calibrated"):
            for split in SPLITS:
                metrics = _target_metrics(
                    y[split], probabilities[model_name][f"{split}_{kind}"],
                )
                metric_rows.append({
                    "model": model_name,
                    "calibration": kind,
                    "split": split,
                    **metrics,
                })
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(report_dir / "pd_metrics.csv", index=False)

    test_predictions = frames["test"][[
        "loan_identifier", "as_of_date", "source_cohort", target,
        "baseline_upb", "classic_fico", "original_ltv", "original_upb",
        "original_dti", "original_interest_rate", "property_state",
        "baseline_loan_age", "anchor_month", "refinance_predecessor_id",
    ]].copy()
    for model_name in ("logistic", "xgboost"):
        test_predictions[f"{model_name}_raw"] = probabilities[model_name]["test_raw"]
        test_predictions[f"{model_name}_calibrated"] = probabilities[model_name]["test_calibrated"]
    test_predictions.to_parquet(
        report_dir / "test_predictions.parquet", index=False, compression="zstd",
    )

    by_year = []
    for model_name in ("logistic", "xgboost"):
        frame = test_predictions.assign(year=test_predictions.as_of_date.dt.year)
        for year, group in frame.groupby("year"):
            by_year.append({
                "model": model_name,
                "year": int(year),
                "rows": len(group),
                "actual_rate": float(group[target].mean()),
                "raw_rate": float(group[f"{model_name}_raw"].mean()),
                "calibrated_rate": float(group[f"{model_name}_calibrated"].mean()),
            })
    pd.DataFrame(by_year).to_csv(report_dir / "pd_test_by_year.csv", index=False)

    deciles = []
    for model_name in ("logistic", "xgboost"):
        frame = pd.DataFrame({
            "p": test_predictions[f"{model_name}_calibrated"],
            "y": test_predictions[target],
        })
        frame["decile"] = pd.qcut(frame.p, 10, labels=False, duplicates="drop") + 1
        grouped = frame.groupby("decile").agg(
            rows=("y", "size"), mean_prediction=("p", "mean"),
            observed_rate=("y", "mean"),
        ).reset_index()
        grouped.insert(0, "model", model_name)
        deciles.append(grouped)
    pd.concat(deciles, ignore_index=True).to_csv(
        report_dir / "pd_calibration_test.csv", index=False,
    )

    coefficients = logistic.named_steps["preprocess"].get_feature_names_out()
    coefficient_frame = pd.DataFrame({
        "feature": coefficients,
        "coefficient": logistic.named_steps["model"].coef_[0],
    })
    coefficient_frame["odds_ratio_per_1sd"] = np.exp(coefficient_frame.coefficient)
    coefficient_frame.sort_values("coefficient", key=np.abs, ascending=False).to_csv(
        report_dir / "logistic_coefficients.csv", index=False,
    )

    importance = xgboost.get_booster().get_score(importance_type="gain")
    pd.DataFrame(
        [{"feature": key, "gain": value} for key, value in importance.items()]
    ).sort_values("gain", ascending=False).to_csv(
        report_dir / "xgboost_importance.csv", index=False,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": logistic,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "calibrator": calibrators["logistic"],
        "train_raw_quantiles": np.quantile(
            probabilities["logistic"]["train_raw"], [0.5, 0.99, 0.995, 0.999],
        ),
    }, model_dir / "logistic.joblib")
    joblib.dump({
        "model": xgboost,
        "levels": levels,
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "calibrator": calibrators["xgboost"],
        "train_raw_quantiles": np.quantile(
            probabilities["xgboost"]["train_raw"], [0.5, 0.99, 0.995, 0.999],
        ),
    }, model_dir / "xgboost.joblib")

    _plot_default_rate(frames, report_dir.parent / "figures" / "default_rate_by_anchor.png")
    _plot_calibration(
        test_predictions,
        report_dir.parent / "figures" / "calibration_test.png",
    )
    return {
        "metrics": metric_frame,
        "test_predictions": test_predictions,
        "calibrators": calibrators,
        "numeric": numeric,
        "categorical": categorical,
    }


def _plot_default_rate(frames: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = []
    for split, frame in frames.items():
        series.append(
            frame.groupby("as_of_date").serious_delinquency_12m.mean()
            .rename(split).reset_index()
        )
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for data in series:
        axis.plot(data.as_of_date, data.iloc[:, 1] * 100, label=data.columns[1])
    axis.set_title("12-month serious-delinquency rate by prediction anchor")
    axis.set_ylabel("Event rate (%)")
    axis.set_xlabel("Prediction anchor")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_calibration(test_predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = {}
    for model_name in ("logistic", "xgboost"):
        curves[model_name] = calibration_curve(
            test_predictions.serious_delinquency_12m,
            test_predictions[f"{model_name}_calibrated"],
            n_bins=12,
            strategy="quantile",
        )
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    for model_name, color in (("logistic", "#476f95"), ("xgboost", "#b35c37")):
        observed, predicted = curves[model_name]
        axis.plot(predicted * 100, observed * 100, marker="o", label=model_name,
                  color=color)
    maximum = max(
        max(predicted.max() for _, predicted in curves.values()) * 100 * 1.1,
        0.1,
    )
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="grey", linewidth=1)
    axis.set_xlim(0, maximum)
    axis.set_ylim(0, maximum)
    axis.set_xlabel("Mean predicted 12-month PD (%)")
    axis.set_ylabel("Observed event rate (%)")
    axis.set_title("2023 out-of-time calibration")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_lgd_training(root: Path):
    outcomes = pd.read_parquet(root / "data" / "modeling" / "lgd_ead_outcomes.parquet")
    feature_paths = sorted((root / "data" / "modeling" / "features").glob("*.parquet"))
    feature_columns = [
        "loan_identifier", "classic_fico", "original_dti",
        "original_interest_rate", "original_ltv", "original_upb", "property_state",
    ]
    features = pd.concat([
        pd.read_parquet(path, columns=feature_columns) for path in feature_paths
    ], ignore_index=True)
    merged = outcomes.merge(features, on="loan_identifier", how="left", validate="one_to_one")
    property_value = merged.original_upb / (merged.original_ltv / 100.0)
    merged["current_ltv_proxy"] = (
        merged.observed_ead / property_value * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    merged["default_year"] = pd.to_datetime(merged.default_date).dt.year
    return merged


def train_lgd(root: Path, config: dict, report_dir: Path, model_dir: Path) -> dict:
    merged = _load_lgd_training(root)
    training = merged[
        merged.use_for_lgd.fillna(False)
        & merged.split.ne("test")
        & merged.observed_loss_fraction.notna()
    ].copy()
    if training.empty:
        raise ValueError("No development LGD observations")

    numeric = ["current_ltv_proxy", "classic_fico", "original_dti",
               "original_interest_rate", "default_year"]
    categorical = ["property_state"]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                      keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                      keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=float(config["lgd"]["ridge_alpha"]), solver="lsqr")),
    ])
    model.fit(
        training[LGD_FEATURES],
        training.observed_loss_fraction.astype(float),
    )
    coefficients = model.named_steps["preprocess"].get_feature_names_out()
    coefficient_frame = pd.DataFrame({
        "feature": coefficients,
        "coefficient": model.named_steps["model"].coef_,
    }).sort_values("coefficient", key=np.abs, ascending=False)
    coefficient_frame.to_csv(report_dir / "lgd_coefficients.csv", index=False)
    current_ltv_row = coefficient_frame[coefficient_frame.feature.eq("numeric__current_ltv_proxy")]
    current_ltv_coefficient = (
        float(current_ltv_row.coefficient.iloc[0]) if not current_ltv_row.empty else None
    )

    availability = pd.read_csv(
        root / "data" / "modeling" / "summaries" / "lgd_availability.csv"
    )
    defaults = int(availability.default_loans.sum())
    resolved = int(availability.resolved_with_ead.sum())
    share_unresolved = 1 - resolved / defaults
    p_liq = resolved / defaults
    eligible = merged[merged.use_for_lgd.fillna(False)]
    current_ltv_correlation = eligible.current_ltv_proxy.corr(
        eligible.observed_loss_fraction
    )
    original_ltv_correlation = eligible.original_ltv.corr(
        eligible.observed_loss_fraction
    )
    info = {
        "default_loans": defaults,
        "resolved_credit_dispositions_with_ead": resolved,
        "p_liq": float(p_liq),
        "share_unresolved": float(share_unresolved),
        "share_serious_delinquency_without_credit_disposition": float(
            share_unresolved
        ),
        "observed_lgd_observations": int(eligible.observed_loss_fraction.notna().sum()),
        "training_observations": len(training),
        "mean_observed_severity": float(eligible.observed_loss_fraction.mean()),
        "median_observed_severity": float(eligible.observed_loss_fraction.median()),
        "share_observed_severity_below_zero": float(
            eligible.observed_loss_fraction.lt(0).mean()
        ),
        "share_observed_severity_above_one": float(
            eligible.observed_loss_fraction.gt(1).mean()
        ),
        "current_ltv_coefficient": current_ltv_coefficient,
        "current_ltv_univariate_correlation": float(current_ltv_correlation),
        "original_ltv_univariate_correlation": float(original_ltv_correlation),
        "current_ltv_coefficient_positive": bool(
            current_ltv_coefficient is not None and current_ltv_coefficient > 0
        ),
        "feature_note": "current_ltv_proxy uses observed EAD and original property value; no HPI path is applied. Its negative conditional and univariate associations are retained and reported, not forced to match the expected sign.",
    }
    _write_json(report_dir / "lgd_info.json", info)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": LGD_FEATURES,
        "numeric": numeric,
        "categorical": categorical,
        "info": info,
    }, model_dir / "lgd.joblib")

    eligible.assign(
        predicted_severity=np.clip(
            model.predict(eligible[LGD_FEATURES]), 0, 1,
        )
    ).to_parquet(
        report_dir / "lgd_observations.parquet", index=False, compression="zstd",
    )
    return {
        "info": info,
        "model": model,
        "features": LGD_FEATURES,
        "merged": merged,
    }


def _predict_lgd(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(bundle["model"].predict(frame[bundle["features"]]), 0, 1)


def _portfolio_lgd_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    property_value = output.original_upb / (output.original_ltv / 100.0)
    output["current_ltv_proxy"] = (
        output.baseline_upb / property_value * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    output["default_year"] = output.as_of_date.dt.year
    return output


def _ecl_summary(frame: pd.DataFrame, pd_column: str) -> dict:
    ecl = frame.ecl
    ead = frame.baseline_upb
    return {
        "loans": int(len(frame)),
        "outstanding_upb": float(ead.sum()),
        "expected_defaults": float(frame[pd_column].sum()),
        "expected_loss": float(ecl.sum()),
        "ecl_rate": float(ecl.sum() / ead.sum()) if ead.sum() else None,
        "mean_pd": float(frame[pd_column].mean()),
        "mean_lgd": float(frame.lgd.mean()),
    }


def _group_ecl(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column, dropna=False).agg(
        loans=("loan_identifier", "size"),
        outstanding_upb=("baseline_upb", "sum"),
        expected_loss=("ecl", "sum"),
        mean_pd=("pd", "mean"),
        mean_lgd=("lgd", "mean"),
    ).reset_index()
    grouped["ecl_rate"] = grouped.expected_loss / grouped.outstanding_upb
    return grouped.rename(columns={column: "band"})


def evaluate_ecl(
    root: Path,
    config: dict,
    report_dir: Path,
    model_dir: Path,
    test_predictions: pd.DataFrame,
    lgd_bundle: dict,
) -> dict:
    portfolio_date = pd.Timestamp(config["ecl"]["portfolio_period"])
    if portfolio_date not in set(test_predictions.as_of_date):
        raise ValueError(f"Portfolio period not present in test set: {portfolio_date}")
    portfolio = _portfolio_lgd_frame(
        test_predictions[test_predictions.as_of_date.eq(portfolio_date)].copy()
    )
    portfolio["lgd"] = _predict_lgd(lgd_bundle, portfolio)
    selected_model = config["ecl"]["pd_model"]
    portfolio["pd"] = portfolio[f"{selected_model}_calibrated"]
    portfolio["ecl"] = portfolio.pd * portfolio.lgd * portfolio.baseline_upb

    fico_bins = [0, 660, 700, 740, 780, 900]
    ltv_bins = [0, 60, 80, 90, 95, 100, 1000]
    portfolio["fico_band"] = pd.cut(
        portfolio.classic_fico, fico_bins, labels=["<660", "660-699", "700-739", "740-779", "780+"],
    )
    portfolio["ltv_band"] = pd.cut(
        portfolio.current_ltv_proxy, ltv_bins,
        labels=["<60", "60-79", "80-89", "90-94", "95-99", "100+"],
    )
    portfolio.to_parquet(
        report_dir / "ecl_portfolio.parquet", index=False, compression="zstd",
    )
    _group_ecl(portfolio, "fico_band").to_csv(report_dir / "ecl_by_fico.csv", index=False)
    _group_ecl(portfolio, "ltv_band").to_csv(report_dir / "ecl_by_ltv.csv", index=False)
    _group_ecl(portfolio, "property_state").to_csv(report_dir / "ecl_by_state.csv", index=False)

    sensitivity = []
    for multiplier in config["ecl"]["sensitivity_pd_multipliers"]:
        for haircut in config["ecl"]["sensitivity_recovery_haircuts"]:
            pd_stressed = np.clip(portfolio.pd * float(multiplier), 0, 1)
            lgd_stressed = np.clip(
                portfolio.lgd + (1 - portfolio.lgd) * float(haircut), 0, 1,
            )
            expected_loss = (pd_stressed * lgd_stressed * portfolio.baseline_upb).sum()
            sensitivity.append({
                "pd_multiplier": float(multiplier),
                "recovery_haircut": float(haircut),
                "ecl_rate": float(expected_loss / portfolio.baseline_upb.sum()),
                "ecl_rate_bps": float(
                    expected_loss / portfolio.baseline_upb.sum() * 10_000
                ),
                "expected_loss": float(expected_loss),
            })
    sensitivity_frame = pd.DataFrame(sensitivity)
    sensitivity_frame.to_csv(report_dir / "sensitivity_grid_bps.csv", index=False)

    scenario_rows = []
    range_rows = []
    for model_name in ("logistic", "xgboost"):
        bundle_path = model_dir / f"{model_name}.joblib"
        bundle = joblib.load(bundle_path)
        quantiles = bundle["train_raw_quantiles"]
        model_portfolio = portfolio.copy()
        model_portfolio["pd"] = model_portfolio[f"{model_name}_calibrated"]
        for scenario_name, scenario in config["ecl"]["scenarios"].items():
            pd_stressed = np.clip(
                model_portfolio.pd * float(scenario["pd_multiplier"]), 0, 1,
            )
            lgd_stressed = np.clip(
                model_portfolio.lgd
                + (1 - model_portfolio.lgd) * float(scenario["recovery_haircut"]),
                0, 1,
            )
            scenario_frame = model_portfolio.assign(
                pd=pd_stressed, lgd=lgd_stressed,
                ecl=pd_stressed * lgd_stressed * model_portfolio.baseline_upb,
            )
            scenario_rows.append({
                "model": model_name,
                "scenario": scenario_name,
                "pd_multiplier": float(scenario["pd_multiplier"]),
                "recovery_haircut": float(scenario["recovery_haircut"]),
                **_ecl_summary(scenario_frame, "pd"),
            })
            range_rows.append({
                "model": model_name,
                "scenario": scenario_name,
                "share_pd_above_train_p99": float(
                    (pd_stressed > quantiles[1]).mean()
                ),
                "share_pd_above_train_p99_5": float(
                    (pd_stressed > quantiles[2]).mean()
                ),
                "share_pd_above_train_p99_9": float(
                    (pd_stressed > quantiles[3]).mean()
                ),
            })
    scenario_frame = pd.DataFrame(scenario_rows)
    scenario_frame.to_csv(report_dir / "stress_scenarios.csv", index=False)
    range_frame = pd.DataFrame(range_rows)
    range_frame.to_csv(report_dir / "stress_range_check.csv", index=False)

    scenario_checks = {}
    for model_name, group in scenario_frame.groupby("model"):
        ordered = group.set_index("scenario").loc[
            ["baseline", "adverse", "severe"]
        ]
        scenario_checks[model_name] = bool(
            ordered.ecl_rate.is_monotonic_increasing
        )
    if not all(scenario_checks.values()):
        raise ValueError(f"Stress ECL is not monotonic: {scenario_checks}")

    backtest = test_predictions.copy()
    backtest["lgd"] = _predict_lgd(lgd_bundle, _portfolio_lgd_frame(backtest))
    backtest["pd"] = backtest[f"{selected_model}_calibrated"]
    backtest["ecl"] = backtest.pd * backtest.lgd * backtest.baseline_upb
    actual = pd.read_parquet(
        root / "data" / "modeling" / "lgd_ead_outcomes.parquet",
        columns=["loan_identifier", "actual_loss", "use_for_lgd"],
    )
    actual = actual[actual.use_for_lgd.fillna(False)][
        ["loan_identifier", "actual_loss"]
    ]
    backtest = backtest.merge(actual, on="loan_identifier", how="left")
    backtest_by_anchor = backtest.groupby("as_of_date").agg(
        loans=("loan_identifier", "size"),
        expected_defaults=("pd", "sum"),
        observed_defaults=("serious_delinquency_12m", "sum"),
        expected_loss=("ecl", "sum"),
        resolved_observed_loss=("actual_loss", "sum"),
        resolved_defaults=("actual_loss", "count"),
    ).reset_index()
    backtest_by_anchor["expected_default_rate"] = (
        backtest_by_anchor.expected_defaults / backtest_by_anchor.loans
    )
    backtest_by_anchor["observed_default_rate"] = (
        backtest_by_anchor.observed_defaults / backtest_by_anchor.loans
    )
    backtest_by_anchor.to_csv(report_dir / "backtest_by_anchor.csv", index=False)
    backtest_summary = {
        "rows": int(len(backtest)),
        "expected_defaults": float(backtest.pd.sum()),
        "observed_defaults": int(backtest.serious_delinquency_12m.sum()),
        "expected_default_ratio": float(
            backtest.pd.sum() / backtest.serious_delinquency_12m.sum()
        ),
        "expected_loss": float(backtest.ecl.sum()),
        "resolved_observed_loss": float(backtest.actual_loss.sum()),
        "resolved_defaults": int(backtest.actual_loss.notna().sum()),
        "note": "Observed loss covers only resolved credit dispositions with disclosed loss; it is not a complete cash-flow back-test.",
    }
    _write_json(report_dir / "backtest_summary.json", backtest_summary)
    _plot_ecl_groups(
        report_dir / "ecl_by_fico.csv",
        "FICO band",
        report_dir.parent / "figures" / "ecl_by_fico.png",
    )
    _plot_ecl_groups(
        report_dir / "ecl_by_ltv.csv",
        "Current LTV band",
        report_dir.parent / "figures" / "ecl_by_ltv.png",
    )
    _plot_scenarios(
        scenario_frame,
        report_dir.parent / "figures" / "stress_scenarios.png",
    )
    _plot_sensitivity(
        sensitivity_frame,
        report_dir.parent / "figures" / "sensitivity_heatmap.png",
    )
    portfolio_summary = {
        **_ecl_summary(portfolio, "pd"),
        "portfolio_period": str(portfolio_date.date()),
        "pd_model": selected_model,
        "scenario_checks": scenario_checks,
    }
    _write_json(report_dir / "portfolio_summary.json", {
        **portfolio_summary,
    })
    return {
        "portfolio": portfolio,
        "summary": portfolio_summary,
        "scenario": scenario_frame,
        "range": range_frame,
        "backtest": backtest_summary,
    }


def _plot_ecl_groups(path: Path, title: str, output: Path) -> None:
    data = pd.read_csv(path)
    data = data[data.band.notna()]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.bar(data.band.astype(str), data.ecl_rate * 100, color="#4f6f7f")
    axis.set_ylabel("ECL rate (% of UPB)")
    axis.set_xlabel(title)
    axis.set_title(f"12-month ECL by {title.lower()}")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_scenarios(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot(index="scenario", columns="model", values="ecl_rate")
    pivot = pivot.loc[["baseline", "adverse", "severe"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    pivot.plot(kind="bar", ax=axis, color=["#526f58", "#b46a48"])
    axis.set_ylabel("ECL rate (% of UPB)")
    axis.set_xlabel("Scenario")
    axis.set_title("12-month ECL under PD and recovery stress")
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_sensitivity(frame: pd.DataFrame, output: Path) -> None:
    pivot = frame.pivot(
        index="pd_multiplier", columns="recovery_haircut", values="ecl_rate",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    image = axis.imshow(pivot.values * 100, aspect="auto", cmap="YlOrBr")
    axis.set_xticks(range(len(pivot.columns)),
                    [f"{value:.0%}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)),
                    [f"{value:.2f}x" for value in pivot.index])
    axis.set_xlabel("Recovery haircut")
    axis.set_ylabel("PD multiplier")
    axis.set_title("ECL rate sensitivity (% of UPB)")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            axis.text(column, row, f"{pivot.values[row, column] * 100:.2f}",
                      ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="ECL rate (%)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_model_report(
    report_dir: Path,
    audit: dict,
    pd_result: dict,
    lgd_result: dict,
    ecl_result: dict,
) -> None:
    metrics = pd_result["metrics"]
    test = metrics[(metrics.split.eq("test")) & metrics.calibration.eq("calibrated")]
    test_table = test[[
        "model", "roc_auc", "ks", "brier", "log_loss",
        "actual_rate", "mean_prediction", "pred_over_actual",
    ]].to_string(index=False, float_format=lambda value: f"{value:.5f}")
    scenario = ecl_result["scenario"]
    scenario_table = scenario[[
        "model", "scenario", "ecl_rate", "expected_loss",
    ]].to_string(index=False, float_format=lambda value: f"{value:.5f}")
    summary = ecl_result["summary"]
    selected_metrics = test[test.model.eq(ecl_result["summary"]["pd_model"])].iloc[0]
    source = audit.get("source_validation", {})
    source_line = (
        f"Sample widths are {source.get('origination_width')} origination and "
        f"{source.get('performance_width')} performance fields; "
        f"{source.get('positive_loss_share', 0):.1%} of signed "
        f"02/03/09 losses are positive."
        if source.get("status") == "pass" else "Raw source archive was not available."
    )
    text = f"""# Freddie Mac PD, LGD and ECL results

## Scope

Release 48 full-vintage handoff: 2008 training, 2018 validation/calibration,
2023 out-of-time test. The sample contains {audit["contract"]["pd_ready_rows"]:,}
labeled loans and {audit["contract"]["feature_count"]} contracting features.
Data files are not tracked in Git.

Source checks: {source_line}

## Out-of-time PD performance

{test_table}

The selected `{ecl_result["summary"]["pd_model"]}` model has
`pred_over_actual = {selected_metrics.pred_over_actual:.3f}`. This is a material
out-of-time calibration error: the model overpredicts the 2023 event rate even
after 2018 calibration. Discrimination is comparatively stable
(`AUC = {selected_metrics.roc_auc:.3f}`), so the result should be used as a
ranking model with explicit scenario haircuts until a newer-period calibration
is available.

## LGD

- Resolved liquidations used for LGD: {lgd_result["info"]["observed_lgd_observations"]:,}
- Mean observed severity: {lgd_result["info"]["mean_observed_severity"]:.4f}
- Current-LTV coefficient positive: {lgd_result["info"]["current_ltv_coefficient_positive"]}
- Current-LTV univariate correlation with severity: {lgd_result["info"]["current_ltv_univariate_correlation"]:.4f}
- Serious-delinquency episodes without an observed 02/03/09 disposition: {lgd_result["info"]["share_serious_delinquency_without_credit_disposition"]:.4f}

The expected positive current-LTV coefficient was not found in this selected
resolution sample. The negative sign is reported rather than forced; it is a
diagnostic that the static LTV proxy and resolution selection dominate the
simple sign expectation.

## Portfolio ECL

Portfolio date: {ecl_result["summary"]["portfolio_period"]}.
Selected PD model: `{ecl_result["summary"]["pd_model"]}`, calibrated on 2018.

- Outstanding UPB: {summary["outstanding_upb"]:,.0f}
- Expected 12-month loss: {summary["expected_loss"]:,.0f}
- ECL rate: {summary["ecl_rate"]:.4%}
- Mean PD: {summary["mean_pd"]:.4%}
- Mean LGD: {summary["mean_lgd"]:.4%}

## Stress scenarios

{scenario_table}

Stress ECL is monotonic from baseline to adverse to severe for both PD models.
Range checks show the share of loans above the training 99.9th percentile PD:
{ecl_result["range"].query("scenario == 'severe'").to_string(index=False)}

## Back-test

Expected defaults across all mature test anchors:
{ecl_result["backtest"]["expected_defaults"]:.1f}; observed defaults:
{ecl_result["backtest"]["observed_defaults"]:,}; ratio:
{ecl_result["backtest"]["expected_default_ratio"]:.3f}. Observed loss covers only
{ecl_result["backtest"]["resolved_defaults"]:,} resolved defaults with a
disclosed loss and is not a complete cash-flow back-test.

## Limitations

This is a 12-month ECL estimate, not lifetime CECL/IFRS 9. Prepayment and cure
are not modeled as competing events in the ECL arithmetic. LGD is selected on
observed resolutions, uses a static current-LTV proxy and is undiscounted.
Scenario probabilities are assumptions. The selected vintages are not a
representative history of all US mortgages.
"""
    (report_dir / "modeling_report.md").write_text(text)


def run_all(root: Path, force: bool = False) -> dict:
    report_root = root / "reports"
    table_dir = report_root / "tables"
    figure_dir = report_root / "figures"
    model_dir = root / "models"
    if report_root.exists() and any(report_root.iterdir()) and not force:
        raise FileExistsError(
            f"{report_root} already contains results; pass --force to replace them"
        )
    if force:
        shutil.rmtree(report_root, ignore_errors=True)
        shutil.rmtree(model_dir, ignore_errors=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(root)

    print("Auditing model contract and split integrity", flush=True)
    audit = audit_data(root, table_dir)
    source_archive = Path(config["source_archive"]).expanduser()
    audit["source_validation"] = validate_source_sample(source_archive)
    _write_json(table_dir / "audit.json", audit)
    pd_result = train_pd(root, config, table_dir, model_dir / "pd")
    print("Fitting LGD", flush=True)
    lgd_result = train_lgd(root, config, table_dir, model_dir / "lgd")
    print("Building ECL and stress results", flush=True)
    ecl_result = evaluate_ecl(
        root, config, table_dir, model_dir / "pd",
        pd_result["test_predictions"], lgd_result,
    )
    _write_model_report(table_dir, audit, pd_result, lgd_result, ecl_result)
    print("Modeling complete", flush=True)
    return {
        "audit": audit,
        "pd": pd_result,
        "lgd": lgd_result,
        "ecl": ecl_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["all", "audit", "train", "ecl"], nargs="?",
        default="all",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "all":
        run_all(root, args.force)
    elif args.command == "audit":
        config = load_config(root)
        audit_data(root, root / "reports" / "tables")
    else:
        if not (root / "reports" / "tables" / "test_predictions.parquet").exists():
            raise FileNotFoundError(
                "Run `python -m src.credit_model all` first; incremental "
                "`train` and `ecl` entry points require saved predictions."
            )
        config = load_config(root)
        if args.command == "train":
            train_pd(root, config, root / "reports" / "tables", root / "models" / "pd")
        else:
            test_predictions = pd.read_parquet(
                root / "reports" / "tables" / "test_predictions.parquet"
            )
            lgd_bundle = joblib.load(root / "models" / "lgd" / "lgd.joblib")
            evaluate_ecl(
                root, config, root / "reports" / "tables",
                root / "models" / "pd", test_predictions, lgd_bundle,
            )


if __name__ == "__main__":
    main()
