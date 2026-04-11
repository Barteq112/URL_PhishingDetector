from __future__ import annotations

import argparse
import ipaddress
import math
import re
from typing import Dict, Iterable, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
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


def load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(csv_path)
    label_col = infer_column(frame.columns, LABEL_CANDIDATES)
    if not label_col:
        raise ValueError(
            "Nie znaleziono kolumny etykiet. Oczekiwane nazwy: "
            + ", ".join(LABEL_CANDIDATES)
        )
    y = encode_labels(frame[label_col])
    X = frame.drop(columns=[label_col]).copy()
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


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if probs is not None:
        result["roc_auc"] = roc_auc_score(y_test, probs)
    else:
        result["roc_auc"] = np.nan

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, digits=4))
    return result


def run_experiment(args: argparse.Namespace) -> None:
    source_X, y = load_data(args.csv)
    X = build_feature_matrix(
        source=source_X,
        use_original_features=not args.no_original_features,
        use_url_features=not args.no_url_features,
    )
    X = pd.get_dummies(X, drop_first=False)
    X_sparse = csr_matrix(X.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
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

    results = [evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test)]

    if args.cv_folds > 1:
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
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
        svm_model = SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=args.random_state)

        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results.append(evaluate_model("Decision Tree", dt_model, X_train, X_test, y_train, y_test))
        results.append(evaluate_model("SVM", svm_model, X_train_scaled, X_test_scaled, y_train, y_test))

    print("\n=== Podsumowanie metryk ===")
    summary = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
    print(summary.to_string(float_format=lambda val: f"{val:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wykrywanie phishingowych URL z uzyciem XGBoost i cech strukturalnych."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Sciezka do pliku CSV z danymi (domyslnie: {DEFAULT_DATA_PATH}).",
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
        help="Rozmiar zbioru testowego (domyslnie 0.2).",
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
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
