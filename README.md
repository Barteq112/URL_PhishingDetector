# URL_PhishingDetector

Modulowy pipeline ekstrakcji cech z URL zbudowany wg schematu:
**parse URL -> domain details -> DNS info -> redirection check -> IP resolution -> character patterns -> auth/keyword detection -> TLD features -> extra features**.

## Struktura
- `main.py` - CLI dla pojedynczego URL i batch CSV
- `url_pipeline\parsing.py` - parsowanie URL (domain/path/query/subdomain/TLD)
- `url_pipeline\domain_features.py` - WHOIS (creation/expiration -> age/validity)
- `url_pipeline\dns_features.py` - MX/SPF/TXT/NS
- `url_pipeline\network_features.py` - redirect check + liczba rozwiazanych IP
- `url_pipeline\lexical_features.py` - wzorce znakow, keywordy, TLD, extra
- `url_pipeline\extractor.py` - orkiestracja calego pipeline

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

4. Tylko cechy lokalne (bez zapytan sieciowych):
```bash
python main.py --url "https://example.com" --no-network
```
