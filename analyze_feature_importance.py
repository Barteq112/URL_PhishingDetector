"""
analyze_feature_importance.py
-----------------------------
Wykres ważności cech wytrenowanego modelu XGBoost (importance_type=gain).

Uruchomienie:
    python analyze_feature_importance.py

Wymaga:
    output/xgboost_model.json
    output/xgboost_metrics_features.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = Path("output/xgboost_model.json")
FEATURES_PATH = Path("output/xgboost_metrics_features.json")
OUTPUT_DIR = Path("analysis/analysis")
DOCS_TOP20 = Path("docs/feature_importance_top20.png")
DOCS_BY_GROUP = Path("docs/feature_importance_by_group.png")
TOP_N = 20

# Kategorie zgodne z sekcją ekstrakcji cech w artykule
FEATURE_GROUPS: dict[str, str] = {
    "port": "struktura",
    "subdomain_count": "struktura",
    "query_param_count": "struktura",
    "query_value_count": "struktura",
    "is_https": "struktura",
    "is_domain_ip": "struktura",
    "has_userinfo": "struktura",
    "has_auth_section": "struktura",
    "subdomain_count_feature": "struktura",
    "is_domain_ip_feature": "struktura",
    "url_length": "wzorce",
    "length_url": "wzorce",
    "domain_length": "wzorce",
    "path_length": "wzorce",
    "query_length": "wzorce",
    "digit_count": "wzorce",
    "letter_count": "wzorce",
    "special_char_count": "wzorce",
    "dot_count": "wzorce",
    "qty_dot_url": "wzorce",
    "hyphen_count": "wzorce",
    "qty_hyphen_path": "wzorce",
    "underscore_count": "wzorce",
    "at_count": "wzorce",
    "slash_count": "wzorce",
    "question_mark_count": "wzorce",
    "equal_count": "wzorce",
    "ampersand_count": "wzorce",
    "percent_count": "wzorce",
    "digit_ratio": "wzorce",
    "special_char_ratio": "wzorce",
    "contains_punycode": "wzorce",
    "has_long_repeated_char_sequence": "wzorce",
    "url_entropy": "entropia",
    "suspicious_keyword_count": "słownik",
    "contains_login_keyword": "słownik",
    "contains_secure_keyword": "słownik",
    "contains_token_keyword": "słownik",
    "has_brand_keyword": "słownik",
    "subdomain_auth_count": "słownik",
    "tld_length": "tld",
    "tld_is_common": "tld",
    "tld_is_suspicious": "tld",
    "is_shortened_url": "tld",
}

GROUP_COLORS = {
    "struktura": "#457B9D",
    "wzorce": "#E76F51",
    "entropia": "#9B5DE5",
    "słownik": "#F4A261",
    "tld": "#2A9D8F",
    "inne": "#6C757D",
}

FEATURE_LABELS_PL: dict[str, str] = {
    "url_entropy": "Entropia URL",
    "url_length": "Długość URL",
    "length_url": "Długość URL (alt.)",
    "domain_length": "Długość domeny",
    "path_length": "Długość ścieżki",
    "query_length": "Długość zapytania",
    "digit_count": "Liczba cyfr",
    "digit_ratio": "Udział cyfr",
    "special_char_count": "Znaki specjalne",
    "special_char_ratio": "Udział znaków spec.",
    "dot_count": "Liczba kropek",
    "qty_dot_url": "Kropki w URL",
    "hyphen_count": "Myślniki",
    "qty_hyphen_path": "Myślniki w ścieżce",
    "subdomain_count": "Liczba subdomen",
    "query_param_count": "Parametry zapytania",
    "query_value_count": "Wartości zapytania",
    "suspicious_keyword_count": "Słowa podejrzane (liczba)",
    "contains_login_keyword": "Słowo: login",
    "contains_secure_keyword": "Słowo: secure",
    "contains_token_keyword": "Słowo: token",
    "has_brand_keyword": "Słowa marek",
    "subdomain_auth_count": "Auth w subdomenie",
    "tld_is_suspicious": "Podejrzane TLD",
    "tld_is_common": "Typowe TLD",
    "tld_length": "Długość TLD",
    "is_shortened_url": "Skracacz URL",
    "is_https": "HTTPS",
    "is_domain_ip": "Domena = IP",
    "has_userinfo": "Dane w URL (userinfo)",
    "has_auth_section": "Sekcja auth",
    "contains_punycode": "Punycode (xn--)",
    "has_long_repeated_char_sequence": "Powtórzenia znaków",
    "port": "Port w URL",
    "at_count": "Znak @",
    "slash_count": "Ukośniki",
    "question_mark_count": "Znak ?",
    "equal_count": "Znak =",
    "ampersand_count": "Znak &",
    "percent_count": "Znak %",
    "letter_count": "Litery",
    "underscore_count": "Podkreślenia",
    "subdomain_count_feature": "Subdomeny (cecha)",
    "is_domain_ip_feature": "IP (cecha)",
}

GROUP_LABELS_PL = {
    "struktura": "Struktura URL",
    "wzorce": "Wzorce znakowe",
    "entropia": "Entropia",
    "słownik": "Słowa kluczowe",
    "tld": "TLD / domena",
    "inne": "Inne",
}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _resolve_feature_name(key: str, feature_names: list[str]) -> str | None:
    if key in feature_names:
        return key
    if key.startswith("f") and key[1:].isdigit():
        idx = int(key[1:])
        if 0 <= idx < len(feature_names):
            return feature_names[idx]
    return None


def load_gain_importance(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    raw = model.get_booster().get_score(importance_type="gain")
    values: dict[str, float] = {name: 0.0 for name in feature_names}

    for key, score in raw.items():
        name = _resolve_feature_name(key, feature_names)
        if name is not None:
            values[name] = float(score)

    rows = []
    for name, gain in values.items():
        group = FEATURE_GROUPS.get(name, "inne")
        rows.append({
            "feature": name,
            "label_pl": FEATURE_LABELS_PL.get(name, name.replace("_", " ")),
            "group": group,
            "group_pl": GROUP_LABELS_PL[group],
            "gain": gain,
        })

    frame = pd.DataFrame(rows)
    total = frame["gain"].sum()
    frame["gain_pct"] = (frame["gain"] / total * 100.0) if total > 0 else 0.0
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def plot_top_features(frame: pd.DataFrame, out_path: Path) -> None:
    top = frame.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [GROUP_COLORS.get(g, GROUP_COLORS["inne"]) for g in top["group"]]

    bars = ax.barh(
        top["label_pl"],
        top["gain"],
        color=colors,
        alpha=0.9,
        zorder=3,
    )
    for bar, pct in zip(bars, top["gain_pct"]):
        w = bar.get_width()
        ax.text(
            w + frame["gain"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Łączny przyrost (gain) w drzewach XGBoost")
    ax.set_title(
        f"Top {TOP_N} cech wpływających na detekcję phishingu",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)

    handles = [
        plt.matplotlib.patches.Patch(color=GROUP_COLORS[k], label=GROUP_LABELS_PL[k])
        for k in ("struktura", "wzorce", "entropia", "słownik", "tld")
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Kategoria cechy")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_group_summary(frame: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        frame.groupby("group", as_index=False)["gain"]
        .sum()
        .sort_values("gain", ascending=True)
    )
    grouped["group_pl"] = grouped["group"].map(GROUP_LABELS_PL)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [GROUP_COLORS.get(g, GROUP_COLORS["inne"]) for g in grouped["group"]]
    ax.barh(grouped["group_pl"], grouped["gain"], color=colors, alpha=0.9)
    total = grouped["gain"].sum()
    for bar, val in zip(ax.patches, grouped["gain"]):
        pct = val / total * 100 if total > 0 else 0
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.0f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Suma gain w grupie")
    ax.set_title("Wpływ kategorii cech na model", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Brak modelu: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Brak listy cech: {FEATURES_PATH}")

    feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list):
        raise ValueError(f"Niepoprawny format: {FEATURES_PATH}")

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    frame = load_gain_importance(model, feature_names)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_top_features(frame, OUTPUT_DIR / "results_04_feature_importance_top20.png")
    plot_group_summary(frame, OUTPUT_DIR / "results_05_feature_importance_by_group.png")

    json_path = OUTPUT_DIR / "feature_importance_gain.json"
    json_path.write_text(
        frame.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    DOCS_TOP20.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_DIR / "results_04_feature_importance_top20.png", DOCS_TOP20)
    shutil.copy2(OUTPUT_DIR / "results_05_feature_importance_by_group.png", DOCS_BY_GROUP)

    print(f"Zapisano wykresy w: {OUTPUT_DIR.resolve()}")
    print(f"Kopia do artykułu: {DOCS_TOP20.resolve()}, {DOCS_BY_GROUP.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print("\nTop 10 cech (gain):")
    for _, row in frame.head(10).iterrows():
        print(f"  {row['label_pl']:32s}  {row['gain']:8.1f}  ({row['gain_pct']:.1f}%)")


if __name__ == "__main__":
    main()
