

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

# ─── Konfiguracja ─────────────────────────────────────────────────────────────

PREDICTIONS = [
    {
        "name": "url_dataset",
        "label": "url\\_dataset",
        "csv": "output/url_dataset_predictions.csv",
    },
    {
        "name": "PhiUSIIL",
        "label": "PhiUSIIL\n(etykiety odwr.)",
        "csv": "output/PhiUSIIL_predictions.csv",
    },
    {
        "name": "data_bal_20000",
        "label": "data\\_bal\n20000",
        "csv": "output/data_bal_predictions.csv",
    },
]

OUTPUT_DIR = Path("analysis/analysis")

COLORS = {
    "url_dataset":    "#2196F3",
    "PhiUSIIL":       "#FF5722",
    "data_bal_20000": "#4CAF50",
}

METRICS_LABELS = {
    "accuracy":  "Dokładność",
    "precision": "Precyzja",
    "recall":    "Czułość",
    "f1":        "F1",
    "roc_auc":   "ROC-AUC",
}

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_predictions(cfg: dict) -> pd.DataFrame | None:
    path = Path(cfg["csv"])
    if not path.exists():
        print(f"  [WARN] Brak pliku: {path} – pomijam.")
        return None
    df = pd.read_csv(path)

    # nazwy kolumn z predict_xgboost.py: label, pred_label, pred_proba
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("pred_label", "predicted", "prediction", "y_pred"):
            rename[col] = "prediction"
        elif cl in ("pred_proba", "prob", "proba", "probability", "score"):
            rename[col] = "probability"
        elif cl in ("true_label", "y_true") and "label" not in df.columns:
            rename[col] = "label"
    df = df.rename(columns=rename)

    required = {"label", "prediction", "probability"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] Brakujące kolumny {missing} w {path.name} – pomijam.")
        print(f"         Dostępne kolumny: {list(df.columns)}")
        return None

    df = df.dropna(subset=["label", "prediction", "probability"])
    print(f"  {cfg['name']:20s}: {len(df):>8,} rekordów")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"].astype(int)
    y_pred = df["prediction"].astype(int)
    y_prob = df["probability"].astype(float)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }

# ─── Wykres 1: Porównanie metryk (grouped bar chart) ─────────────────────────

def plot_metrics_comparison(
    results: dict[str, dict],
    out: Path,
) -> None:
    metrics  = list(METRICS_LABELS.keys())
    n_groups = len(metrics)
    n_sets   = len(results)
    width    = 0.22
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (name, vals) in enumerate(results.items()):
        offset = (i - n_sets / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            [vals[m] for m in metrics],
            width,
            label=name,
            color=COLORS[name],
            alpha=0.88,
            zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=6.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Wartość metryki")
    ax.set_title(
        "Porównanie metryk klasyfikacji na trzech zbiorach danych",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}")
    )

    fig.tight_layout()
    fig.savefig(out / "results_01_metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] results_01_metrics_comparison.png")

# ─── Wykres 2: Krzywe ROC ─────────────────────────────────────────────────────

def plot_roc_curves(
    frames: dict[str, pd.DataFrame],
    results: dict[str, dict],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))

    ax.plot([0, 1], [0, 1], color="lightgray", lw=1.2,
            linestyle="--", label="Losowy klasyfikator (AUC = 0.50)")

    for name, df in frames.items():
        y_true = df["label"].astype(int)
        y_prob = df["probability"].astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = results[name]["roc_auc"]
        ax.plot(
            fpr, tpr,
            color=COLORS[name],
            lw=2.2,
            label=f"{name}  (AUC = {auc:.4f})",
        )

    ax.set_xlabel("Odsetek fałszywych alarmów (FPR)")
    ax.set_ylabel("Czułość (TPR)")
    ax.set_title("Krzywe ROC dla trzech zbiorów danych",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25)

    # zaznacz punkt (0,1) – idealny klasyfikator
    ax.plot(0, 1, marker="*", color="gold", markersize=12, zorder=5,
            label="Idealny klasyfikator")

    fig.tight_layout()
    fig.savefig(out / "results_02_roc_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] results_02_roc_curves.png")

# ─── Wykres 3: Macierze konfuzji ──────────────────────────────────────────────

def plot_confusion_matrices(
    frames: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, frames.items()):
        y_true = df["label"].astype(int)
        y_pred = df["prediction"].astype(int)
        cm = confusion_matrix(y_true, y_pred)

        total = cm.sum()
        cm_pct = cm / total * 100

        color = COLORS[name]
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.2)

        for i in range(2):
            for j in range(2):
                val_abs = cm[i, j]
                val_pct = cm_pct[i, j]
                text_color = "white" if cm_pct[i, j] > cm_pct.max() * 0.6 else "black"
                ax.text(j, i,
                        f"{val_abs:,}\n({val_pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=text_color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legalne (pred.)", "Phishing (pred.)"], fontsize=8)
        ax.set_yticklabels(["Legalne (rzecz.)", "Phishing (rzecz.)"], fontsize=8)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8, color=color)
        ax.set_xlabel("Przewidziana klasa", fontsize=9)
        ax.set_ylabel("Rzeczywista klasa", fontsize=9)

    fig.suptitle("Macierze konfuzji dla trzech zbiorów danych",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "results_03_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] results_03_confusion_matrices.png")

# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Wczytywanie predykcji...")
    frames: dict[str, pd.DataFrame] = {}
    for cfg in PREDICTIONS:
        df = load_predictions(cfg)
        if df is not None:
            frames[cfg["name"]] = df

    if len(frames) < 2:
        print("[ERROR] Za mało plików predykcji (minimum 2). Sprawdź ścieżki w PREDICTIONS.")
        return

    print("\n[2/4] Obliczanie metryk...")
    results: dict[str, dict] = {}
    for name, df in frames.items():
        m = compute_metrics(df)
        results[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  roc_auc={m['roc_auc']:.4f}")

    print("\n[3/4] Generowanie wykresów...")
    plot_metrics_comparison(results, OUTPUT_DIR)
    plot_roc_curves(frames, results, OUTPUT_DIR)
    plot_confusion_matrices(frames, OUTPUT_DIR)

    print(f"\nGotowe! Wykresy zapisane w: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()