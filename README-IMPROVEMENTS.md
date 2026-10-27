# Improved ML evaluation and evidence verification

Copy these files into your existing project, replacing same-named files:
- `requirements.txt`
- `src/prepare_data.py`
- `src/train.py`
- `app/app.py`

Add:
- `src/verify_claims.py`

Keep your existing `src/preprocess.py`.

## Install
```powershell
pip install -r requirements.txt
```

## Rebuild dataset
The updated preparation script preserves subject/date metadata for analysis only:
```powershell
python -m src.prepare_data
```
This overwrites `data/news.csv`, not the original `Fake.csv` and `True.csv`.

## Train and evaluate
```powershell
python -m src.train --data data/news.csv
```
This reports random holdout metrics and attempts a chronological holdout when dates are usable. It compares five classifiers and saves the selected pipeline and metrics.

Chronological testing is not an unseen-publisher test. The dataset lacks reliable publisher identifiers, so random-split results may be optimistic due to source/topic patterns.

## Launch
```powershell
streamlit run app/app.py
```

The app has tabs for text classification, evidence retrieval, and model evaluation.

## Optional Google Fact Check Tools API
The web evidence search works without a key. To also query Google's Fact Check Tools API, configure a key for the current PowerShell session:
```powershell
$env:GOOGLE_FACT_CHECK_API_KEY = "YOUR_API_KEY"
streamlit run app/app.py
```
Do not put your key in source code or share it.

## Important
The evidence module retrieves search results and existing fact-check reviews. It does not infer truth from keyword overlap or snippets. A human should open sources and compare names, numbers, dates, and context.
