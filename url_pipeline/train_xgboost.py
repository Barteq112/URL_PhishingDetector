from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

LABEL_MAP = {
    "phishing": 1,
    "malicious": 1,
    "legitimate": 0,
    "benign": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trenowanie modelu XGBoost na pliku CSV z cechami.")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV z cechami i etykieta.")
    parser.add_argument("--label-column", type=str, default="label", help="Nazwa kolumny etykiety.")
    parser.add_argument("--output-model", type=str, default="output\\xgboost_model.json", help="Plik modelu.")
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="output\\xgboost_metrics.json",
        help="Plik z metrykami.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Procent danych testowych (0-1).")
    parser.add_argument("--random-state", type=int, default=42, help="Seed losowania.")
    parser.add_argument("--bayes-iter", type=int, default=100, help="Budzet ewaluacji BayesSearchCV.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Liczba foldow Stratified CV.")
    return parser.parse_args()


def _normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        return int(float(text))
    except ValueError:
        raise ValueError(f"Nieznana etykieta: {value!r}") from None


def train_xgboost(
    input_csv: str,
    label_column: str,
    output_model: str,
    output_metrics: str,
    test_size: float,
    random_state: int,
    bayes_iter: int,
    cv_folds: int,
) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    if label_column not in frame.columns:
        raise ValueError(f"Brak kolumny etykiety '{label_column}' w pliku: {input_csv}")

    y = frame[label_column].apply(_normalize_label)
    if y.nunique() < 2:
        raise ValueError("Model wymaga co najmniej 2 klas w kolumnie label.")

    x_raw = frame.drop(columns=[label_column])
    x = x_raw.select_dtypes(include=["number", "bool"]).copy()
    x = x.fillna(0)

    if x.shape[1] == 0:
        raise ValueError("Brak cech numerycznych do trenowania modelu.")

    stratify = y if y.nunique() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    search_spaces = {
        "n_estimators": Integer(100, 1200),
        "mah": Integer(3, 12),
        "learning_x_deptrate": Real(0.01, 0.3, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "min_child_weight": Integer(1, 20),
        "gamma": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_lambda": Real(1e-8, 10.0, prior="log-uniform"),
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    tuner = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=bayes_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    tuner.fit(x_train, y_train)
    model = tuner.best_estimator_

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, object] = {
        "rows_total": int(len(frame)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "features_used": int(x.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "bayes_best_cv_score": float(tuner.best_score_),
        "bayes_iter": int(bayes_iter),
        "cv_folds": int(cv_folds),
        "n_jobs": int(-1),
    }
    metrics["best_params"] = {key: value for key, value in tuner.best_params_.items()}

    model_path = Path(output_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    metrics_path = Path(output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    features_path = metrics_path.with_name(metrics_path.stem + "_features.json")
    features_path.write_text(json.dumps(list(x.columns), ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    args = parse_args()
    metrics = train_xgboost(
        input_csv=args.input_csv,
        label_column=args.label_column,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        test_size=args.test_size,
        random_state=args.random_state,
        bayes_iter=args.bayes_iter,
        cv_folds=args.cv_folds,
    )
    print(f"Model zapisany do: {args.output_model}")
    print(f"Metryki zapisane do: {args.output_metrics}")
    print(
        "Wynik: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
