from pathlib import Path
import joblib
from src.preprocess import clean_text
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models'/'fake_news_pipeline.joblib'

def predict(text):
    if not MODEL.exists():
        raise FileNotFoundError('Train first: python -m src.train --data data/news.csv')
    cleaned = clean_text(text)
    if not cleaned: raise ValueError('News text cannot be empty.')
    return joblib.load(MODEL).predict([cleaned])[0]

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('text'); args=p.parse_args()
    print('Predicted class:', predict(args.text))
    print('Reminder: this is a model prediction, not a verified fact-check.')
