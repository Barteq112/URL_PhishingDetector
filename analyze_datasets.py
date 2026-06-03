"""
analyze_datasets.py
-------------------
Analiza pokrycia i właściwości zbiorów danych URL do wykrywania phishingu.

Uruchomienie:
    python analyze_datasets.py

Wymagania:
    pip install pandas matplotlib seaborn tldextract

Generuje pliki:
    output/analysis/01_overlap_venn.png
    output/analysis/02_label_distribution.png
    output/analysis/03_url_length_distribution.png
    output/analysis/04_tld_top20.png
    output/analysis/05_overlap_heatmap.png
    output/analysis/dataset_report.txt
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    print("[WARN] tldextract niedostępny – analiza TLD zostanie pominięta.")

# ─── Konfiguracja ─────────────────────────────────────────────────────────────

DATASETS = [
    {
        "name": "url_dataset",
        "input_csv": "data/url_dataset.csv",
        "url_column": "url",
        "label_column": "type",
    },
    {
        "name": "PhiUSIIL",
        "input_csv": "data/PhiUSIIL_Phishing_URL_Dataset.csv",
        "url_column": "URL",
        "label_column": "label",
    },
    {
        "name": "data_bal_20000",
        "input_csv": "data/data_bal - 20000.csv",
        "url_column": "URLs",
        "label_column": "Labels",
    },
]

LABEL_MAP = {
    "phishing": 1, "malicious": 1,
    "legitimate": 0, "benign": 0,
}

OUTPUT_DIR = Path("analysis/Dataset_analysis")

# Paleta kolorów
COLORS = {
    "url_dataset":    "#E63946",
    "PhiUSIIL":       "#457B9D",
    "data_bal_20000": "#2A9D8F",
    "phishing":       "#E76F51",
    "legitimate":     "#52B788",
    "overlap":        "#F4A261",
}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        v = int(float(text))
        return v if v in (0, 1) else -1
    except ValueError:
        return -1


def load_dataset(cfg: dict) -> pd.DataFrame | None:
    path = Path(cfg["input_csv"])
    if not path.exists():
        print(f"[WARN] Plik nie istnieje: {path} – pomijam.")
        return None
    df = pd.read_csv(path)
    if cfg["url_column"] not in df.columns or cfg["label_column"] not in df.columns:
        print(f"[WARN] Brak kolumn w {path.name} – pomijam.")
        return None
    out = pd.DataFrame()
    out["url"]   = df[cfg["url_column"]].astype(str).str.strip()
    out["label"] = df[cfg["label_column"]].apply(normalize_label)
    out = out[(out["url"] != "") & (out["label"] != -1)].drop_duplicates(subset=["url"])
    out["source"] = cfg["name"]
    print(f"  {cfg['name']:20s}: {len(out):>7,} rekordów  "
          f"(phishing={( out['label']==1).sum():,}  "
          f"legit={(out['label']==0).sum():,})")
    return out


def extract_domain(url: str) -> str:
    if HAS_TLDEXTRACT:
        ext = tldextract.extract(url)
        return ext.registered_domain or ext.domain or url
    # fallback – wyciągnij coś między // a pierwszym /
    try:
        return url.split("//")[-1].split("/")[0].split("?")[0].lower()
    except Exception:
        return url


def extract_tld(url: str) -> str:
    if HAS_TLDEXTRACT:
        return tldextract.extract(url).suffix or "unknown"
    try:
        domain = url.split("//")[-1].split("/")[0]
        parts = domain.split(".")
        return parts[-1] if len(parts) > 1 else "unknown"
    except Exception:
        return "unknown"


# ─── Wykresy ──────────────────────────────────────────────────────────────────

def plot_label_distribution(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Słupkowy wykres rozkładu klas w każdym zbiorze."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names   = list(frames.keys())
    x       = np.arange(len(names))
    width   = 0.35

    phishing = [( frames[n]["label"] == 1).sum() for n in names]
    legit    = [( frames[n]["label"] == 0).sum() for n in names]

    b1 = ax.bar(x - width/2, phishing, width, label="Phishing (1)",
                color=COLORS["phishing"], alpha=0.9, zorder=3)
    b2 = ax.bar(x + width/2, legit,    width, label="Legalne (0)",
                color=COLORS["legitimate"], alpha=0.9, zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(phishing + legit)*0.01,
                f"{h:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Liczba rekordów")
    ax.set_title("Rozkład klas w każdym zbiorze danych", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out / "02_label_distribution.png")
    plt.close(fig)
    print("  [OK] 02_label_distribution.png")


def plot_url_length(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Rozkład długości URL (histogram + KDE) dla każdego zbioru."""
    fig, axes = plt.subplots(1, len(frames), figsize=(5*len(frames), 4), sharey=False)
    if len(frames) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, frames.items()):
        lengths = df["url"].str.len().clip(upper=300)
        color   = COLORS.get(name, "#888")

        ax.hist(lengths[df["label"] == 1], bins=60, alpha=0.6,
                color=COLORS["phishing"], label="Phishing", density=True)
        ax.hist(lengths[df["label"] == 0], bins=60, alpha=0.6,
                color=COLORS["legitimate"], label="Legalne", density=True)

        med_p = lengths[df["label"] == 1].median()
        med_l = lengths[df["label"] == 0].median()
        ax.axvline(med_p, color=COLORS["phishing"],   linestyle="--", lw=1.5,
                   label=f"Mediana phish: {med_p:.0f}")
        ax.axvline(med_l, color=COLORS["legitimate"], linestyle="--", lw=1.5,
                   label=f"Mediana legit: {med_l:.0f}")

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Długość URL (max 300)")
        ax.set_ylabel("Gęstość")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Rozkład długości URL wg klasy", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "03_url_length_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 03_url_length_distribution.png")


def plot_tld_top20(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Top-20 TLD osobno dla phishingu i legalnych (łącznie ze wszystkich zbiorów)."""
    combined = pd.concat(frames.values(), ignore_index=True)

    phish_tlds = Counter(combined[combined["label"] == 1]["url"].apply(extract_tld))
    legit_tlds = Counter(combined[combined["label"] == 0]["url"].apply(extract_tld))

    top_p = pd.DataFrame(phish_tlds.most_common(20), columns=["tld", "count"])
    top_l = pd.DataFrame(legit_tlds.most_common(20), columns=["tld", "count"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(top_p["tld"][::-1], top_p["count"][::-1],
             color=COLORS["phishing"], alpha=0.85)
    ax1.set_title("Top 20 TLD – Phishing", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Liczba URL")
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(top_l["tld"][::-1], top_l["count"][::-1],
             color=COLORS["legitimate"], alpha=0.85)
    ax2.set_title("Top 20 TLD – Legalne", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Liczba URL")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Najczęstsze domeny najwyższego poziomu (TLD)", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "04_tld_top20.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 04_tld_top20.png")


def plot_overlap_heatmap(sets: dict[str, set], frames: dict[str, pd.DataFrame],
                         out: Path) -> None:
    """Macierz pokrycia: ile URL-i ze zbioru A jest w zbiorze B (w %)."""
    names = list(sets.keys())
    n = len(names)
    mat_abs = np.zeros((n, n), dtype=int)
    mat_pct = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            overlap = len(sets[ni] & sets[nj])
            mat_abs[i, j] = overlap
            mat_pct[i, j] = overlap / len(sets[ni]) * 100 if sets[ni] else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat_pct, cmap="YlOrRd", vmin=0, vmax=100)

    for i in range(n):
        for j in range(n):
            val_pct = mat_pct[i, j]
            val_abs = mat_abs[i, j]
            color = "white" if val_pct > 55 else "black"
            ax.text(j, i, f"{val_pct:.1f}%\n({val_abs:,})",
                    ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_title("Macierz pokrycia URL\n(wiersz = źródło, kolumna = cel; % URL z wiersza obecnych w kolumnie)",
                 fontsize=10, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, label="Pokrycie [%]")
    fig.tight_layout()
    fig.savefig(out / "05_overlap_heatmap.png")
    plt.close(fig)
    print("  [OK] 05_overlap_heatmap.png")


def plot_venn3(sets: dict[str, set], out: Path) -> None:
    """Ręczny diagram Venna dla 3 zbiorów (bez biblioteki matplotlib-venn)."""
    names = list(sets.keys())
    s = [sets[n] for n in names]

    only    = [len(s[i] - s[j] - s[k]) for i, j, k in [(0,1,2),(1,0,2),(2,0,1)]]
    pair_01 = len(s[0] & s[1] - s[2])
    pair_02 = len(s[0] & s[2] - s[1])
    pair_12 = len(s[1] & s[2] - s[0])
    all3    = len(s[0] & s[1] & s[2])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.set_aspect("equal")
    ax.axis("off")

    circles = [
        plt.Circle((3.8, 5.5), 2.5, color=COLORS["url_dataset"],    alpha=0.30, zorder=2),
        plt.Circle((6.2, 5.5), 2.5, color=COLORS["PhiUSIIL"],       alpha=0.30, zorder=2),
        plt.Circle((5.0, 3.3), 2.5, color=COLORS["data_bal_20000"], alpha=0.30, zorder=2),
    ]
    for c in circles:
        ax.add_patch(c)

    # Etykiety zbiorów
    label_positions = [(1.4, 7.6), (8.6, 7.6), (5.0, 0.6)]
    colors_list = [COLORS["url_dataset"], COLORS["PhiUSIIL"], COLORS["data_bal_20000"]]
    for (lx, ly), name, col in zip(label_positions, names, colors_list):
        total = len(sets[name])
        ax.text(lx, ly, f"{name}\n({total:,})", ha="center", va="center",
                fontsize=9, fontweight="bold", color=col)

    # Wartości w obszarach
    texts = [
        (2.5, 6.2, f"{only[0]:,}"),
        (7.5, 6.2, f"{only[1]:,}"),
        (5.0, 2.2, f"{only[2]:,}"),
        (5.0, 6.6, f"{pair_01:,}"),
        (3.5, 4.0, f"{pair_02:,}"),
        (6.5, 4.0, f"{pair_12:,}"),
        (5.0, 5.0, f"{all3:,}"),
    ]
    for tx, ty, txt in texts:
        ax.text(tx, ty, txt, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Diagram Venna – pokrycie URL między zbiorami danych",
                 fontsize=13, fontweight="bold", pad=16)

    legend_patches = [mpatches.Patch(color=c, alpha=0.6, label=n)
                      for n, c in zip(names, colors_list)]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "01_overlap_venn.png", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 01_overlap_venn.png")


def save_report(frames: dict[str, pd.DataFrame],
                sets: dict[str, set], out: Path) -> None:
    """Zapisz raport tekstowy z kluczowymi statystykami."""
    names = list(frames.keys())
    lines = []
    lines.append("=" * 62)
    lines.append("RAPORT ANALIZY ZBIORÓW DANYCH – URL PHISHING DETECTOR")
    lines.append("=" * 62)

    lines.append("\n--- Statystyki per zbiór danych ---\n")
    for name, df in frames.items():
        total   = len(df)
        phish   = (df["label"] == 1).sum()
        legit   = (df["label"] == 0).sum()
        ratio   = phish / total * 100 if total else 0
        med_len = df["url"].str.len().median()
        avg_len = df["url"].str.len().mean()
        lines.append(f"  {name}")
        lines.append(f"    Rekordy łącznie : {total:>8,}")
        lines.append(f"    Phishing        : {phish:>8,}  ({ratio:.1f}%)")
        lines.append(f"    Legalne         : {legit:>8,}  ({100-ratio:.1f}%)")
        lines.append(f"    Mediana dł. URL : {med_len:>8.1f} znaków")
        lines.append(f"    Średnia dł. URL : {avg_len:>8.1f} znaków")
        lines.append("")

    lines.append("--- Pokrycie par zbiorów ---\n")
    for (n1, n2) in combinations(names, 2):
        inter  = len(sets[n1] & sets[n2])
        union  = len(sets[n1] | sets[n2])
        jaccard = inter / union * 100 if union else 0
        lines.append(f"  {n1}  ∩  {n2}")
        lines.append(f"    Wspólne URL     : {inter:>8,}")
        lines.append(f"    Jaccard [%]     : {jaccard:>8.2f}%")
        lines.append("")

    all3 = len(sets[names[0]] & sets[names[1]] & sets[names[2]]) if len(names) >= 3 else 0
    total_unique = len(set.union(*sets.values()))
    lines.append("--- Suma wszystkich zbiorów ---\n")
    lines.append(f"  Unikalne URL łącznie      : {total_unique:>8,}")
    lines.append(f"  Wspólne we wszystkich 3   : {all3:>8,}")

    report_path = out / "dataset_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] dataset_report.txt")

    # Wydruk na konsolę
    print()
    print("\n".join(lines))


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Wczytywanie zbiorów danych...")
    frames: dict[str, pd.DataFrame] = {}
    for cfg in DATASETS:
        df = load_dataset(cfg)
        if df is not None:
            frames[cfg["name"]] = df

    if len(frames) < 2:
        print("[ERROR] Za mało zbiorów danych do analizy (minimum 2).")
        return

    print("\n[2/4] Obliczanie pokrycia URL...")
    sets: dict[str, set] = {name: set(df["url"]) for name, df in frames.items()}
    for name, s in sets.items():
        print(f"  {name}: {len(s):,} unikalnych URL")

    print("\n[3/4] Generowanie wykresów...")
    plot_venn3(sets, OUTPUT_DIR)
    plot_label_distribution(frames, OUTPUT_DIR)
    plot_url_length(frames, OUTPUT_DIR)
    if HAS_TLDEXTRACT:
        plot_tld_top20(frames, OUTPUT_DIR)
    else:
        print("  [SKIP] 04_tld_top20.png (brak tldextract)")
    plot_overlap_heatmap(sets, frames, OUTPUT_DIR)

    print("\n[4/4] Zapisywanie raportu tekstowego...")
    save_report(frames, sets, OUTPUT_DIR)

    print(f"\nGotowe! Wykresy zapisane w: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
