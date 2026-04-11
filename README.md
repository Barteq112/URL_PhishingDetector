# URL_PhishingDetector

Projekt wykrywania phishingowych adresow URL oparty o **XGBoost**, z mozliwoscia porownania wynikow z **Decision Tree** i **SVM**.

## Zakres
- przygotowanie danych (lokalny CSV albo automatyczne pobranie UCI dataset ID 967),
- ekstrakcja cech URL (dlugosc, znaki specjalne, IP w host, TLD itd.),
- trening i walidacja modelu XGBoost,
- metryki: accuracy, precision, recall, F1, ROC-AUC,
- opcjonalne porownanie z prostszymi klasyfikatorami.

## Wymagania
```bash
pip install -r requirements.txt
```

## Uruchomienie
1. Trening XGBoost na UCI dataset:
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

## Wymagany format danych CSV
- kolumna etykiety o nazwie jednej z: `label`, `class`, `target`, `result`, `status`, `phishing`,
- opcjonalnie kolumna URL o nazwie jednej z: `url`, `uri`, `link`, `domain`, `website`.

Jesli URL nie ma, model uzyje dostepnych cech liczbowych.
