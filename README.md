# URL_PhishingDetector

Projekt wykrywania phishingowych adresow URL oparty o **XGBoost**, z mozliwoscia porownania wynikow z **Decision Tree** i **SVM**.

## Zakres
- przygotowanie danych z pliku CSV,
- ekstrakcja cech URL (dlugosc, znaki specjalne, IP w host, TLD itd.),
- podzial danych na zbiory: uczacy, walidacyjny i testowy,
- trening, walidacja i zapis modelu XGBoost,
- metryki: accuracy, precision, recall, F1, ROC-AUC,
- opcjonalne porownanie z prostszymi klasyfikatorami,
- wczytanie zapisanego modelu i predykcja bez ponownego uczenia,
- automatyczne generowanie wykresow ewaluacyjnych.

## Wymagania
```bash
pip install -r requirements.txt
```

## Uruchomienie
1. Trening XGBoost (uczenie + walidacja + test + zapis modelu):
```bash
python main.py
```

2. Trening + porownanie z Decision Tree i SVM:
```bash
python main.py --compare
```

3. Uzycie wlasnego pliku CSV:
```bash
python main.py --csv sciezka_do_pliku.csv --compare
```

Domyslny plik: `data/PhiUSIIL_Phishing_URL_Dataset.csv`

4. Predykcja na podstawie zapisanego modelu (bez uczenia):
```bash
python main.py --mode predict --model-path models\xgboost_phishing_model.pkl --csv data\PhiUSIIL_Phishing_URL_Dataset.csv --predictions-out predictions.csv
```

Domyslny model: `models\xgboost_phishing_model.pkl`

5. Wlasny podzial i katalog wykresow:
```bash
python main.py --test-size 0.2 --val-size 0.2 --plots-dir plots --compare
```

Generowane wykresy (domyslnie w `plots\`):
- `xgboost_confusion_matrix_test.png`
- `xgboost_roc_curve_test.png`
- `xgboost_pr_curve_test.png`
- `xgboost_feature_importance.png`

## Wymagany format danych CSV
- kolumna etykiety o nazwie jednej z: `label`, `class`, `target`, `result`, `status`, `phishing`,
- opcjonalnie kolumna URL o nazwie jednej z: `url`, `uri`, `link`, `domain`, `website`.

Jesli URL nie ma, model uzyje dostepnych cech liczbowych.
