from __future__ import annotations

import argparse
import ipaddress
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SUSPICIOUS_TLDS = {
    "zip",
    "review",
    "country",
    "kim",
    "cricket",
    "science",
    "work",
    "click",
    "gq",
    "ml",
    "cf",
    "tk",
    "ga",
}

LABEL_CANDIDATES = ("label", "class", "target", "result", "status", "phishing")
URL_CANDIDATES = ("url", "uri", "link", "domain", "website")
DEFAULT_DATA_PATH = "data\\PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_MODEL_PATH = "models\\xgboost_phishing_model.pkl"
DEFAULT_PLOTS_DIR = "plots"


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    probs = [value.count(ch) / len(value) for ch in set(value)]
    return -sum(p * math.log2(p) for p in probs)


def normalize_url(url: str) -> str:
    candidate = str(url).strip()
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate


def host_is_ip(hostname: str) -> int:
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def extract_url_features(url: str) -> Dict[str, float]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    tld = host.split(".")[-1].lower() if "." in host else ""
    full = str(url)

    return {
        "url_length": len(full),
        "host_length": len(host),
        "path_length": len(parsed.path or ""),
        "query_length": len(parsed.query or ""),
        "digit_count": sum(ch.isdigit() for ch in full),
        "special_char_count": len(re.findall(r"[^A-Za-z0-9]", full)),
        "dot_count": full.count("."),
        "hyphen_count": full.count("-"),
        "slash_count": full.count("/"),
        "at_count": full.count("@"),
        "question_mark_count": full.count("?"),
        "equal_count": full.count("="),
        "ampersand_count": full.count("&"),
        "has_ip_host": host_is_ip(host),
        "uses_https": 1 if parsed.scheme == "https" else 0,
        "subdomain_count": max(host.count(".") - 1, 0) if host else 0,
        "tld_length": len(tld),
        "tld_is_suspicious": 1 if tld in SUSPICIOUS_TLDS else 0,
        "url_entropy": shannon_entropy(full.lower()),
    }


def infer_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def encode_labels(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    mapping = {
        "phishing": 1,
        "malicious": 1,
        "bad": 1,
        "legitimate": 0,
        "benign": 0,
        "good": 0,
        "safe": 0,
    }
    normalized = y.astype(str).str.strip().str.lower()
    if normalized.isin(mapping.keys()).all():
        return normalized.map(mapping).astype(int)

    unique_values = sorted(normalized.unique().tolist())
    if len(unique_values) != 2:
        raise ValueError("Dataset musi zawierac klasy binarne.")

    inferred = {unique_values[0]: 0, unique_values[1]: 1}
    return normalized.map(inferred).astype(int)


def load_data(csv_path: str, require_label: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
    frame = pd.read_csv(csv_path)
    label_col = infer_column(frame.columns, LABEL_CANDIDATES)
    if require_label and not label_col:
        raise ValueError(
            "Nie znaleziono kolumny etykiet. Oczekiwane nazwy: "
            + ", ".join(LABEL_CANDIDATES)
        )
    y = encode_labels(frame[label_col]) if label_col else None
    X = frame.drop(columns=[label_col]).copy() if label_col else frame.copy()
    return X, y


def build_feature_matrix(
    source: pd.DataFrame, use_original_features: bool, use_url_features: bool
) -> pd.DataFrame:
    features = pd.DataFrame(index=source.index)

    if use_original_features:
        numeric = source.select_dtypes(include=[np.number]).copy()
        if not numeric.empty:
            features = pd.concat([features, numeric], axis=1)

    if use_url_features:
        url_col = infer_column(source.columns, URL_CANDIDATES)
        if url_col:
            url_features = source[url_col].astype(str).apply(extract_url_features).apply(pd.Series)
            features = pd.concat([features, url_features], axis=1)
        else:
            print(
                "UWAGA: Nie znaleziono kolumny URL, pomijam cechy URL "
                f"(oczekiwane: {', '.join(URL_CANDIDATES)})."
            )

    if features.empty:
        raise ValueError("Brak cech do trenowania modelu.")

    return features.fillna(0.0)


def evaluate_trained_model(name: str, model, X_test, y_test) -> Dict[str, object]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if probs is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
    else:
        metrics["roc_auc"] = np.nan

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    return {"metrics": metrics, "predictions": preds, "probabilities": probs}


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, output_path: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model: XGBClassifier, feature_columns: list[str], output_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title("Top 20 Feature Importances (XGBoost)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model: XGBClassifier,
    feature_columns: list[str],
    plots_dir: str,
) -> None:
    output_dir = Path(plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        y_true,
        y_pred,
        output_dir / "xgboost_confusion_matrix_test.png",
        "XGBoost - Confusion Matrix (Test)",
    )
    if y_prob is not None:
        plot_roc_curve(
            y_true,
            y_prob,
            output_dir / "xgboost_roc_curve_test.png",
            "XGBoost - ROC Curve (Test)",
        )
        plot_pr_curve(
            y_true,
            y_prob,
            output_dir / "xgboost_pr_curve_test.png",
            "XGBoost - Precision-Recall Curve (Test)",
        )
    plot_feature_importance(model, feature_columns, output_dir / "xgboost_feature_importance.png")


def save_model_artifact(
    model_path: str,
    model: XGBClassifier,
    feature_columns: list[str],
    use_original_features: bool,
    use_url_features: bool,
) -> None:
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "use_original_features": use_original_features,
        "use_url_features": use_url_features,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(artifact, file)


def load_model_artifact(model_path: str) -> dict:
    with Path(model_path).open("rb") as file:
        return pickle.load(file)


def align_to_training_columns(features: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    aligned = features.reindex(columns=training_columns, fill_value=0.0)
    return aligned


def run_experiment(args: argparse.Namespace) -> None:
    if args.mode == "train":
        source_X, y = load_data(args.csv, require_label=True)
        X = build_feature_matrix(
            source=source_X,
            use_original_features=not args.no_original_features,
            use_url_features=not args.no_url_features,
        )
        X = pd.get_dummies(X, drop_first=False)
        X_sparse = csr_matrix(X.values)

        if args.test_size <= 0 or args.val_size <= 0:
            raise ValueError("test-size i val-size musza byc > 0.")
        if args.test_size + args.val_size >= 1:
            raise ValueError("Suma test-size i val-size musi byc < 1.")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X_sparse,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        val_relative_size = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_relative_size,
            random_state=args.random_state,
            stratify=y_temp,
        )
        print(
            f"Podzial danych: uczacy={X_train.shape[0]}, "
            f"walidacyjny={X_val.shape[0]}, testowy={X_test.shape[0]}"
        )

        xgb_model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.random_state,
            n_jobs=-1,
        )

        xgb_model.fit(X_train, y_train)
        val_eval = evaluate_trained_model("XGBoost (walidacja)", xgb_model, X_val, y_val)
        test_eval = evaluate_trained_model("XGBoost (test)", xgb_model, X_test, y_test)
        results = [val_eval["metrics"], test_eval["metrics"]]

        save_model_artifact(
            model_path=args.model_path,
            model=xgb_model,
            feature_columns=X.columns.tolist(),
            use_original_features=not args.no_original_features,
            use_url_features=not args.no_url_features,
        )
        print(f"Model zapisany do: {args.model_path}")

        if not args.no_plots:
            generate_plots(
                y_true=y_test,
                y_pred=np.asarray(test_eval["predictions"]),
                y_prob=np.asarray(test_eval["probabilities"]) if test_eval["probabilities"] is not None else None,
                model=xgb_model,
                feature_columns=X.columns.tolist(),
                plots_dir=args.plots_dir,
            )
            print(f"Wykresy zapisane do: {args.plots_dir}")

        if args.cv_folds > 1:
            splitter = StratifiedKFold(
                n_splits=args.cv_folds, shuffle=True, random_state=args.random_state
            )
            scores = cross_val_score(xgb_model, X_sparse, y, cv=splitter, scoring="f1", n_jobs=-1)
            print(
                f"XGBoost {args.cv_folds}-fold CV F1: "
                f"{scores.mean():.4f} (+/- {scores.std():.4f})"
            )

        if args.compare:
            dt_model = DecisionTreeClassifier(
                max_depth=18,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=args.random_state,
            )
            svm_model = SVC(
                C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=args.random_state
            )

            scaler = StandardScaler(with_mean=False)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_val_scaled = scaler.transform(X_val)

            dt_model.fit(X_train, y_train)
            svm_model.fit(X_train_scaled, y_train)

            results.append(evaluate_trained_model("Decision Tree (walidacja)", dt_model, X_val, y_val)["metrics"])
            results.append(evaluate_trained_model("Decision Tree (test)", dt_model, X_test, y_test)["metrics"])
            results.append(
                evaluate_trained_model("SVM (walidacja)", svm_model, X_val_scaled, y_val)["metrics"]
            )
            results.append(evaluate_trained_model("SVM (test)", svm_model, X_test_scaled, y_test)["metrics"])

        print("\n=== Podsumowanie metryk ===")
        summary = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
        print(summary.to_string(float_format=lambda val: f"{val:.4f}"))
        return

    artifact = load_model_artifact(args.model_path)
    source_X, y = load_data(args.csv, require_label=False)
    features = build_feature_matrix(
        source=source_X,
        use_original_features=artifact["use_original_features"],
        use_url_features=artifact["use_url_features"],
    )
    features = pd.get_dummies(features, drop_first=False)
    features = align_to_training_columns(features, artifact["feature_columns"])
    X_sparse = csr_matrix(features.values)

    model = artifact["model"]
    probs = model.predict_proba(X_sparse)[:, 1]
    preds = model.predict(X_sparse)

    output = source_X.copy()
    output["predicted_label"] = preds
    output["predicted_phishing_probability"] = probs
    output.to_csv(args.predictions_out, index=False)
    print(f"Predykcje zapisane do: {args.predictions_out}")

    if y is not None:
        evaluate_trained_model("XGBoost (wczytany model)", model, X_sparse, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wykrywanie phishingowych URL z uzyciem XGBoost i cech strukturalnych."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "predict"),
        default="train",
        help="Tryb dzialania: train (uczenie i zapis) albo predict (wczytanie i predykcja).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Sciezka do pliku CSV z danymi (domyslnie: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Sciezka modelu do zapisu/odczytu (domyslnie: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="predictions.csv",
        help="Plik wyjsciowy z predykcjami dla trybu predict.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Porownaj XGBoost z Decision Tree i SVM.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Rozmiar zbioru testowego jako udzial calego zbioru (domyslnie 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Rozmiar zbioru walidacyjnego jako udzial calego zbioru (domyslnie 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed losowosci (domyslnie 42).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Liczba foldow cross-validation dla XGBoost. Ustaw 0 lub 1, aby pominac.",
    )
    parser.add_argument(
        "--no-original-features",
        action="store_true",
        help="Pomin oryginalne cechy liczbowe z datasetu.",
    )
    parser.add_argument(
        "--no-url-features",
        action="store_true",
        help="Pomin cechy wyciagane bezposrednio z URL.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help=f"Katalog wyjsciowy na wykresy (domyslnie: {DEFAULT_PLOTS_DIR}).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Wylacz generowanie wykresow.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
