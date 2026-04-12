# URL_PhishingDetector

Pipeline ekstrakcji cech z URL zbudowany wg schematu:
**parse URL -> character patterns -> auth/keyword detection -> TLD features -> extra features**.

## Struktura
- `main.py` - CLI dla pojedynczego URL i batch CSV
- `normalize_csv.py` - prosty skrypt do ujednolicenia CSV
- `url_pipeline\extractor.py` - caly pipeline ekstrakcji cech w jednym pliku

## Instalacja
```bash
pip install -r requirements.txt
```

## Uzycie
1. Pojedynczy URL (wynik JSON na stdout):
```bash
python main.py --url "https://example.com/login?token=123"
```

2. Pojedynczy URL z zapisem do pliku:
```bash
python main.py --url "https://example.com/login?token=123" --output output\features.json
```

3. Batch z CSV:
```bash
python main.py --input-csv data\PhiUSIIL_Phishing_URL_Dataset.csv --url-column URL --output output\features.csv
```

3a. Ujednolicenie kazdego CSV osobno:
```bash
python normalize_csv.py
```

Skrypt ma reczna liste `DATASETS` w `normalize_csv.py` - tam ustawiasz:
- `input_csv`
- `url_column`
- `label_column`
- `output_csv`

Kazdy dataset zapisuje sie do osobnego pliku (kolumny: `URL,label`).

