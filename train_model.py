import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_and_prep_data():
    print(" Loading data... (this might take a few seconds)")

    events = pd.read_csv('data_resources/events.zip')
    ginf = pd.read_csv('data_resources/ginf.csv')

    print(f"Original events: {len(events)}")
    shots = events[events['event_type'] == 1].copy()
    print(f"Shots only: {len(shots)}")

    ginf_cols = ['id_odsp', 'season', 'league', 'country']
    shots = shots.merge(ginf[ginf_cols], on='id_odsp', how='left')

    return shots


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    shots = load_and_prep_data()

    print(" Cleaning text...")
    shots['clean_text'] = shots['text'].apply(clean_text)

    print(" Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(shots['clean_text'])
    y = shots['is_goal']

    print(" Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(" Model Trained!")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print(" Saving files for Dashboard...")
    joblib.dump(model, 'xg_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

    cols_to_keep = [
        'id_odsp', 'sort_order', 'time', 'text', 'event_type',
        'player', 'player2', 'side', 'event_team', 'opponent',
        'is_goal', 'shot_place', 'shot_outcome', 'location',
        'bodypart', 'assist_method', 'situation', 'season'
    ]
    shots[cols_to_keep].to_csv('shots_data.csv', index=False)

    print("Done! All files are saved.")

if __name__ == "__main__":
    main()