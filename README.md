# URL_PhishingDetector

Pipeline ekstrakcji cech z URL zbudowany wg schematu:
**parse URL -> character patterns -> auth/keyword detection -> TLD features -> extra features**.

## Struktura
- `main.py` - pusty plik (tymczasowo)
- `normalize_csv.py` - prosty skrypt do ujednolicenia CSV
- `url_pipeline\extractor.py` - caly pipeline ekstrakcji cech + CLI
- `url_pipeline\train_xgboost.py` - modul trenowania modelu XGBoost
- `url_pipeline\predict_xgboost.py` - uruchamianie wytrenowanego modelu (bez uczenia)

## Instalacja
```bash
pip install -r requirements.txt
```

## Uzycie
1. Pojedynczy URL (wynik JSON na stdout):
```bash
python -m url_pipeline.extractor --url "https://example.com/login?token=123"
```

2. Pojedynczy URL z zapisem do pliku:
```bash
python -m url_pipeline.extractor --url "https://example.com/login?token=123" --output output\features.json
```

3. Batch z CSV:
```bash
python -m url_pipeline.extractor --input-csv data\PhiUSIIL_Phishing_URL_Dataset.csv --url-column URL --output output\features.csv
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

4. Trenowanie XGBoost:
```bash
python -m url_pipeline.train_xgboost --input-csv output\url_dataset_features.csv --label-column label --output-model output\xgboost_model.json --output-metrics output\xgboost_metrics.json
```

Trening uzywa BayesSearchCV (scikit-optimize): stratified 5-fold CV, budzet 100 ewaluacji, `n_jobs=-1`.
Mozesz zmienic parametry:
```bash
python -m url_pipeline.train_xgboost --input-csv output\url_dataset_features.csv --label-column label --bayes-iter 100 --cv-folds 5
```

5. Predykcja wytrenowanym modelem (bez uczenia):
```bash
python -m url_pipeline.predict_xgboost --input-csv output\url_dataset_features.csv --model-path output\xgboost_model.json --features-file output\xgboost_metrics_features.json --label-column label --output-csv output\url_dataset_predictions.csv
```

Jesli w danych jest kolumna etykiety (`label-column`), skrypt wypisze dokladnosc predykcji (accuracy, precision, recall, f1, roc_auc).


