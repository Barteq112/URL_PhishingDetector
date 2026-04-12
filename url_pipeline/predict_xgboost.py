from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

LABEL_MAP = {
    "phishing": 1,
    "malicious": 1,
    "legitimate": 0,
    "benign": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchomienie wytrenowanego modelu XGBoost (bez uczenia).")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV z cechami do predykcji.")
    parser.add_argument("--model-path", type=str, required=True, help="Sciezka do modelu .json.")
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Kolumna etykiety do policzenia metryk (jesli istnieje).",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="JSON z lista cech (np. output\\xgboost_metrics_features.json).",
    )
    parser.add_argument("--output-csv", type=str, required=True, help="Plik wynikowy z predykcjami.")
    return parser.parse_args()


def _normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        return int(float(text))
    except ValueError:
        raise ValueError(f"Nieznana etykieta: {value!r}") from None


def _prepare_features(frame: pd.DataFrame, features_file: str | None, label_column: str) -> pd.DataFrame:
    frame_no_label = frame.drop(columns=[label_column], errors="ignore")
    if features_file:
        feature_names = json.loads(Path(features_file).read_text(encoding="utf-8"))
        if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
            raise ValueError(f"Niepoprawny plik cech: {features_file}")

        x = frame_no_label.copy()
        for feature in feature_names:
            if feature not in x.columns:
                x[feature] = 0
        x = x[feature_names]
    else:
        x = frame_no_label.select_dtypes(include=["number", "bool"]).copy()

    if x.shape[1] == 0:
        raise ValueError("Brak cech numerycznych do predykcji.")

    return x.fillna(0)


def predict_xgboost(
    input_csv: str,
    model_path: str,
    output_csv: str,
    features_file: str | None = None,
    label_column: str = "label",
) -> tuple[int, dict[str, float] | None]:
    frame = pd.read_csv(input_csv)
    x = _prepare_features(frame, features_file, label_column)

    model = XGBClassifier()
    model.load_model(model_path)

    pred_label = model.predict(x)
    pred_proba = model.predict_proba(x)[:, 1]

    out = frame.copy()
    out["pred_label"] = pred_label.astype(int)
    out["pred_proba"] = pred_proba.astype(float)

    metrics: dict[str, float] | None = None
    if label_column in frame.columns:
        y_true = frame[label_column].apply(_normalize_label)
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred_label)),
            "precision": float(precision_score(y_true, pred_label, zero_division=0)),
            "recall": float(recall_score(y_true, pred_label, zero_division=0)),
            "f1": float(f1_score(y_true, pred_label, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, pred_proba)),
        }

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return len(out), metrics


def main() -> None:
    args = parse_args()
    rows, metrics = predict_xgboost(
        input_csv=args.input_csv,
        model_path=args.model_path,
        output_csv=args.output_csv,
        features_file=args.features_file,
        label_column=args.label_column,
    )
    print(f"Zapisano predykcje dla {rows} rekordow do: {args.output_csv}")
    if metrics:
        print(
            "Dokladnosc predykcji: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
