from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from url_pipeline.extractor import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ekstrakcji cech URL (parse -> domain -> dns -> redirect -> ip -> pattern -> auth/keywords -> tld -> extra)."
    )
    parser.add_argument("--url", type=str, default=None, help="Pojedynczy URL do analizy.")
    parser.add_argument("--input-csv", type=str, default=None, help="Plik CSV z URL-ami.")
    parser.add_argument(
        "--url-column",
        type=str,
        default="URL",
        help="Nazwa kolumny URL w pliku CSV (domyslnie: URL).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Sciezka pliku wyjsciowego (JSON dla --url, CSV dla --input-csv).",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Pomin cechy sieciowe (WHOIS, DNS, redirect, IP resolution).",
    )
    return parser.parse_args()


def run_single_url(url: str, output: str | None, enable_network: bool) -> None:
    features = extract_features(url=url, enable_network=enable_network)
    payload = json.dumps(features, ensure_ascii=False, indent=2)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Zapisano cechy do: {out_path}")
        return
    print(payload)


def run_csv(input_csv: str, url_column: str, output: str | None, enable_network: bool) -> None:
    frame = pd.read_csv(input_csv)
    if url_column not in frame.columns:
        raise ValueError(f"Brak kolumny '{url_column}' w pliku: {input_csv}")

    features_frame = frame[url_column].astype(str).apply(
        lambda value: pd.Series(extract_features(value, enable_network=enable_network))
    )
    merged = pd.concat([frame, features_frame], axis=1)

    destination = output or "features_output.csv"
    out_path = Path(destination)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Zapisano cechy dla {len(frame)} URL-i do: {out_path}")


def main() -> None:
    args = parse_args()
    if not args.url and not args.input_csv:
        raise ValueError("Podaj --url albo --input-csv.")
    if args.url and args.input_csv:
        raise ValueError("Uzyj tylko jednego trybu: --url albo --input-csv.")

    enable_network = not args.no_network
    if args.url:
        run_single_url(url=args.url, output=args.output, enable_network=enable_network)
        return
    run_csv(
        input_csv=args.input_csv,
        url_column=args.url_column,
        output=args.output,
        enable_network=enable_network,
    )


if __name__ == "__main__":
    main()
