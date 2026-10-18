"""Train the regression, classification, and clustering models.

Every comparison uses the same held-out farms. Preprocessing is fit inside a
Pipeline on the training farms only. Yield, yield per hectare, and high_risk
are never model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    LLLL,
    REGRESSION_TARGET,
    VERIFICATION_CODE,
)
from src.preprocess import make_preprocessor
from src.validate import assert_no_target_leakage

RANDOM_FOREST_SETTINGS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 3,
    "random_state": LLLL,
    "n_jobs": 1,
}


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )


def _positive_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the probability of high_risk=1 using the fitted class order."""
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in set(classes):
        raise ValueError("The classifier never saw high_risk=1 in training, so a probability cannot be reported.")
    return pipeline.predict_proba(features)[:, classes.index(1)]


def _regression_scores(actual, predicted) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _classification_scores(actual, predicted, probability) -> dict[str, object]:
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(actual, probability))
    except ValueError:
        roc_auc = None
    return {
        "precision": float(precision_score(actual, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "roc_auc": roc_auc,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def train_regression(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, object]:
    """Compare a mean baseline with linear regression and a random forest."""
    assert_no_target_leakage(list(FEATURE_COLUMNS))
    features_train = train_frame[FEATURE_COLUMNS]
    features_test = test_frame[FEATURE_COLUMNS]
    actual_train = train_frame[REGRESSION_TARGET]
    actual_test = test_frame[REGRESSION_TARGET]

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(**RANDOM_FOREST_SETTINGS),
    }
    candidate_rows = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(features_train, actual_train)
        predicted = pipeline.predict(features_test)
        row = {"name": name, **_regression_scores(actual_test, predicted)}
        if name == "random_forest":
            row["hyperparameters"] = RANDOM_FOREST_SETTINGS
        else:
            row["hyperparameters"] = {"fit_intercept": True}
        candidate_rows.append(row)
        fitted[name] = pipeline

    baseline = _pipeline(DummyRegressor(strategy="mean"))
    baseline.fit(features_train, actual_train)
    baseline_predicted = baseline.predict(features_test)
    baseline_scores = {"name": "mean_baseline", "strategy": "mean", **_regression_scores(actual_test, baseline_predicted)}

    selected_name = min(candidate_rows, key=lambda row: (row["rmse"], row["mae"], -row["r2"]))["name"]
    selected = fitted[selected_name]
    selected_predicted = selected.predict(features_test)
    selected_scores = next(row for row in candidate_rows if row["name"] == selected_name)
    reason = (
        f"{selected_name} has the lowest held-out RMSE "
        f"({selected_scores['rmse']:.4f} tons), compared with the mean baseline RMSE "
        f"of {baseline_scores['rmse']:.4f} tons."
    )
    return {
        "baseline": baseline_scores,
        "candidates": candidate_rows,
        "selected_model": selected_name,
        "selection_reason": reason,
        "pipeline": selected,
        "test_prediction": selected_predicted,
    }


def train_classification(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, object]:
    """Compare a majority baseline with logistic regression and a random forest."""
    assert_no_target_leakage(list(FEATURE_COLUMNS))
    features_train = train_frame[FEATURE_COLUMNS]
    features_test = test_frame[FEATURE_COLUMNS]
    actual_train = train_frame[CLASSIFICATION_TARGET].astype(int)
    actual_test = test_frame[CLASSIFICATION_TARGET].astype(int)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=LLLL),
        "random_forest": RandomForestClassifier(**RANDOM_FOREST_SETTINGS),
    }
    candidate_rows = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = _pipeline(estimator)
        pipeline.fit(features_train, actual_train)
        predicted = pipeline.predict(features_test)
        probability = _positive_probability(pipeline, features_test)
        row = {"name": name, "positive_class": 1, **_classification_scores(actual_test, predicted, probability)}
        if name == "random_forest":
            row["hyperparameters"] = RANDOM_FOREST_SETTINGS
        else:
            row["hyperparameters"] = {"max_iter": 2000, "random_state": LLLL}
        candidate_rows.append(row)
        fitted[name] = (pipeline, predicted, probability)

    baseline = _pipeline(DummyClassifier(strategy="most_frequent"))
    baseline.fit(features_train, actual_train)
    baseline_predicted = baseline.predict(features_test)
    baseline_probability = _positive_probability(baseline, features_test)
    baseline_scores = {
        "name": "majority_baseline",
        "strategy": "most_frequent",
        "positive_class": 1,
        **_classification_scores(actual_test, baseline_predicted, baseline_probability),
    }

    selected_name = max(
        candidate_rows,
        key=lambda row: (row["f1"], -1.0 if row["roc_auc"] is None else row["roc_auc"], row["recall"]),
    )["name"]
    selected, selected_predicted, selected_probability = fitted[selected_name]
    selected_scores = next(row for row in candidate_rows if row["name"] == selected_name)
    tied = [
        row
        for row in candidate_rows
        if row["name"] != selected_name and round(float(row["f1"]), 6) == round(float(selected_scores["f1"]), 6)
    ]
    if tied:
        other = tied[0]
        reason = (
            f"{selected_name} ties {other['name']} on held-out F1 for high_risk=1 "
            f"({selected_scores['f1']:.4f}). It is selected because its ROC-AUC is higher "
            f"({selected_scores['roc_auc']:.4f} versus {other['roc_auc']:.4f}). "
            f"The majority baseline F1 is {baseline_scores['f1']:.4f}."
        )
    else:
        reason = (
            f"{selected_name} has the highest held-out F1 for high_risk=1 "
            f"({selected_scores['f1']:.4f}), compared with the majority baseline F1 "
            f"of {baseline_scores['f1']:.4f}."
        )
    return {
        "baseline": baseline_scores,
        "candidates": candidate_rows,
        "selected_model": selected_name,
        "selection_reason": reason,
        "pipeline": selected,
        "test_prediction": selected_predicted,
        "test_probability": selected_probability,
    }


def _cluster_profile(train_frame: pd.DataFrame, labels: np.ndarray, cluster_id: int) -> dict[str, object]:
    members = train_frame.iloc[np.flatnonzero(np.asarray(labels) == cluster_id)]
    overall = train_frame
    numeric_columns = [
        "rainfall_mm",
        "avg_temp_c",
        "soil_moisture_pct",
        "fertilizer_kg",
        "field_size_ha",
    ]
    means = {}
    for column in numeric_columns:
        value = members[column].mean(skipna=True)
        if pd.notna(value):
            means[column] = float(value)
    contrasts = []
    descriptions = {
        "rainfall_mm": ("rainfall", "mm", 0),
        "avg_temp_c": ("temperature", "C", 1),
        "soil_moisture_pct": ("soil moisture", "%", 1),
        "fertilizer_kg": ("fertilizer use", "kg", 1),
        "field_size_ha": ("field size", "ha", 2),
    }
    for column, (label, unit, digits) in descriptions.items():
        spread = float(overall[column].std(ddof=0))
        if spread <= 1e-9:
            continue
        z_score = (means[column] - float(overall[column].mean())) / spread
        if z_score >= 0.35:
            contrasts.append(f"higher {label} ({means[column]:.{digits}f} {unit})")
        elif z_score <= -0.35:
            contrasts.append(f"lower {label} ({means[column]:.{digits}f} {unit})")

    support_rate = float(members["cooperative_support"].astype(float).mean())
    overall_support = float(overall["cooperative_support"].astype(float).mean())
    if support_rate >= overall_support + 0.10:
        contrasts.append(f"more farms receiving cooperative support ({support_rate:.0%})")
    elif support_rate <= overall_support - 0.10:
        contrasts.append(f"fewer farms receiving cooperative support ({support_rate:.0%})")

    season_a_share = float((members["season"] == "A").mean())
    if season_a_share >= 0.60:
        season_text = "mostly season A"
    elif season_a_share <= 0.40:
        season_text = "mostly season B"
    else:
        season_text = "a mix of season A and season B"

    if contrasts:
        contrast_text = ", ".join(contrasts)
    else:
        contrast_text = "growing conditions close to the training-set average"
    support_only = len(contrasts) == 1 and "cooperative support" in contrasts[0]
    if support_only:
        meaning = (
            "Rainfall, temperature, soil moisture, fertilizer, and field size stay near the training average, "
            "so this group is defined by support status rather than by a different climate or field size. "
            "A field officer can use it to tell supported farms from unsupported farms."
        )
    else:
        meaning = (
            "A field officer can plan one visit pattern for these farms because they share these growing conditions."
        )
    summary = (
        f"Cluster {cluster_id} contains {len(members)} training farms with {contrast_text}. "
        f"The group is {season_text}. {meaning} Yield and risk were not used to form the group."
    )
    return {
        "cluster": int(cluster_id),
        "training_farms": int(len(members)),
        "means": {column: round(value, 4) for column, value in means.items()},
        "season_a_share": round(season_a_share, 4),
        "cooperative_support_rate": round(support_rate, 4),
        "profile": summary,
    }


def train_clustering(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, object]:
    """Choose K from 2 to 6 with the silhouette score, then profile each cluster."""
    assert_no_target_leakage(list(FEATURE_COLUMNS))
    features_train = train_frame[FEATURE_COLUMNS]
    features_test = test_frame[FEATURE_COLUMNS]
    search_preprocessor = make_preprocessor()
    transformed = search_preprocessor.fit_transform(features_train)

    scores = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=LLLL, n_init=10)
        labels = model.fit_predict(transformed)
        score = float(silhouette_score(transformed, labels))
        scores.append({"k": k, "silhouette": score})

    selected = max(scores, key=lambda row: (row["silhouette"], -row["k"]))
    selected_k = int(selected["k"])
    pipeline = _pipeline(KMeans(n_clusters=selected_k, random_state=LLLL, n_init=10))
    pipeline.fit(features_train)
    train_labels = pipeline.predict(features_train)
    test_labels = pipeline.predict(features_test)
    profiles = [_cluster_profile(train_frame, train_labels, cluster_id) for cluster_id in range(selected_k)]
    reason = (
        f"K={selected_k} has the highest silhouette score ({selected['silhouette']:.4f}) "
        f"among K=2, 3, 4, 5, and 6. The score was calculated on training farms after "
        f"imputation, scaling, and season encoding fit on the training split only."
    )
    if selected["silhouette"] < 0.25:
        reason += " The separation is modest, so the groups overlap and should be treated as a rough organising aid."

    centers = pipeline.named_steps["model"].cluster_centers_
    return {
        "k_scores": scores,
        "selected_k": selected_k,
        "selection_reason": reason,
        "fit_on": "training_split_only",
        "scaled_centers": centers.round(6).tolist(),
        "profiles": profiles,
        "pipeline": pipeline,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "transformed_train": pipeline.named_steps["preprocessor"].transform(features_train),
    }


def build_test_predictions(test_frame: pd.DataFrame, regression: dict, classification: dict, clustering: dict) -> pd.DataFrame:
    """Assemble the held-out comparison table used by the report."""
    actual_yield = test_frame[REGRESSION_TARGET].to_numpy(dtype=float)
    predicted_yield = np.asarray(regression["test_prediction"], dtype=float)
    table = pd.DataFrame(
        {
            "row_id": test_frame["row_id"].astype(int),
            "actual_yield_tons": actual_yield,
            "predicted_yield_tons": predicted_yield,
            "yield_error": actual_yield - predicted_yield,
            "actual_high_risk": test_frame[CLASSIFICATION_TARGET].astype(int),
            "predicted_high_risk": np.asarray(classification["test_prediction"]).astype(int),
            "high_risk_probability": np.asarray(classification["test_probability"], dtype=float),
            "cluster": np.asarray(clustering["test_labels"]).astype(int),
        }
    )
    return table.sort_values("row_id").reset_index(drop=True)


def notable_regression_error(test_frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    """Describe the largest absolute yield error on the held-out farms."""
    ordered = predictions.assign(absolute_error=lambda frame: frame["yield_error"].abs())
    ordered = ordered.sort_values(["absolute_error", "row_id"], ascending=[False, True])
    row = ordered.iloc[0]
    farm = test_frame.loc[test_frame["row_id"] == int(row["row_id"])].iloc[0]
    direction = "overestimates" if row["yield_error"] < 0 else "underestimates"
    note = (
        f"Row {int(row['row_id'])} has the largest absolute test error. "
        f"The selected model {direction} harvest by {abs(float(row['yield_error'])):.4f} tons "
        f"(actual {float(row['actual_yield_tons']):.4f}, predicted {float(row['predicted_yield_tons']):.4f}). "
        f"That farm has {float(farm['field_size_ha']):.4f} ha, {float(farm['rainfall_mm']):.1f} mm of rainfall, "
        f"and season {farm['season']}."
    )
    return {
        "row_id": int(row["row_id"]),
        "actual_yield_tons": float(row["actual_yield_tons"]),
        "predicted_yield_tons": float(row["predicted_yield_tons"]),
        "yield_error": float(row["yield_error"]),
        "note": note,
    }


def notable_classification_error(test_frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    """Describe the most confident high-risk mistake on the held-out farms."""
    actual = predictions["actual_high_risk"].to_numpy()
    predicted = predictions["predicted_high_risk"].to_numpy()
    probability = predictions["high_risk_probability"].to_numpy()
    row_ids = predictions["row_id"].to_numpy()
    false_negatives = np.flatnonzero((actual == 1) & (predicted == 0))
    false_positives = np.flatnonzero((actual == 0) & (predicted == 1))
    if len(false_negatives):
        choice = false_negatives[np.lexsort((row_ids[false_negatives], probability[false_negatives]))[0]]
        error_type = "false_negative"
    elif len(false_positives):
        choice = false_positives[np.lexsort((row_ids[false_positives], -probability[false_positives]))[0]]
        error_type = "false_positive"
    else:
        choice = int(np.lexsort((row_ids, np.abs(probability - 0.5)))[0])
        error_type = "none"

    row_id = int(row_ids[choice])
    farm = test_frame.loc[test_frame["row_id"] == row_id].iloc[0]
    if error_type == "false_negative":
        note = (
            f"Row {row_id} is a missed high-risk farm. The model predicted low risk with "
            f"high-risk probability {float(probability[choice]):.4f}. Actual yield is "
            f"{float(farm[REGRESSION_TARGET]):.4f} tons on {float(farm['field_size_ha']):.4f} ha."
        )
    elif error_type == "false_positive":
        note = (
            f"Row {row_id} was flagged high risk, but the actual label is low risk. "
            f"The high-risk probability was {float(probability[choice]):.4f}. Actual yield is "
            f"{float(farm[REGRESSION_TARGET]):.4f} tons on {float(farm['field_size_ha']):.4f} ha."
        )
    else:
        note = "The selected classifier made no mistakes on this held-out split."
    return {
        "row_id": row_id,
        "error_type": error_type,
        "actual_high_risk": int(actual[choice]),
        "predicted_high_risk": int(predicted[choice]),
        "high_risk_probability": float(probability[choice]),
        "note": note,
    }


def train_all(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> dict[str, object]:
    """Fit all three tasks and return metrics, pipelines, and test predictions."""
    regression = train_regression(train_frame, test_frame)
    classification = train_classification(train_frame, test_frame)
    clustering = train_clustering(train_frame, test_frame)
    predictions = build_test_predictions(test_frame, regression, classification, clustering)
    regression["notable_test_error"] = notable_regression_error(test_frame, predictions)
    classification["notable_test_error"] = notable_classification_error(test_frame, predictions)
    return {
        "verification_code": VERIFICATION_CODE,
        "regression": regression,
        "classification": classification,
        "clustering": clustering,
        "predictions": predictions,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
    }
