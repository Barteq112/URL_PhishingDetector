from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "phishing": 1,
    "malicious": 1,
    "legitimate": 0,
    "benign": 0,
}

# Ustaw tu recznie pliki i kolumny.
DATASETS = [
    {
        "input_csv": "data\\url_dataset.csv",
        "url_column": "url",
        "label_column": "type",
        "output_csv": "data\\normalized\\url_dataset_normalized.csv",
    },
    {
        "input_csv": "data\\PhiUSIIL_Phishing_URL_Dataset.csv",
        "url_column": "URL",
        "label_column": "label",
        "output_csv": "data\\normalized\\PhiUSIIL_Phishing_URL_Dataset_normalized.csv",
    },
    {
        "input_csv": "data\\data_bal - 20000.csv",
        "url_column": "URLs",
        "label_column": "Labels",
        "output_csv": "data\\normalized\\data_bal-20000.csv_normalized.csv",
    },
]

def _normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]
    try:
        return int(float(text))
    except ValueError:
        return -1


def normalize_one_csv(input_csv: str, url_column: str, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    if url_column not in frame.columns:
        raise ValueError(f"Brak kolumny '{url_column}' w pliku: {input_csv}")
    if label_column not in frame.columns:
        raise ValueError(f"Brak kolumny '{label_column}' w pliku: {input_csv}")

    normalized = pd.DataFrame()
    normalized["URL"] = frame[url_column].astype(str).str.strip()
    normalized["label"] = frame[label_column].apply(_normalize_label)
    return normalized[normalized["URL"] != ""].drop_duplicates(subset=["URL"])


def main() -> None:
    if not DATASETS:
        raise ValueError("Lista DATASETS jest pusta.")

    for dataset in DATASETS:
        normalized = normalize_one_csv(
            input_csv=dataset["input_csv"],
            url_column=dataset["url_column"],
            label_column=dataset["label_column"],
        )
        out_path = Path(dataset["output_csv"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(out_path, index=False)
        print(f"Zapisano {len(normalized)} rekordow do: {out_path}")


if __name__ == "__main__":
    main()
