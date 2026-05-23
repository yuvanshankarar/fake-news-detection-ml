# ============================================
# FAKE NEWS DETECTION PROJECT
# Using Machine Learning (Logistic Regression)
# ============================================

import argparse
import io
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET_PATH = r"C:\Users\artwi\Desktop\Fake News Detection\dataset.csv"


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n" \
            f"Please place your CSV file in the project folder or pass --dataset <path>."
        )

    with open(dataset_path, 'r', encoding='utf-8') as dataset_file:
        raw = dataset_file.read()

    if raw.strip().startswith('|'):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        separator_pattern = re.compile(r'^\|?\s*[-: ]+\s*\|')
        if len(lines) >= 2 and separator_pattern.match(lines[1]):
            cleaned_lines = []
            for line in lines:
                if separator_pattern.match(line):
                    continue
                cleaned_lines.append(re.sub(r'^\||\|$', '', line).strip())

            cleaned = '\n'.join(cleaned_lines)
            df = pd.read_csv(
                io.StringIO(cleaned),
                sep=r'\s*\|\s*',
                engine='python',
                on_bad_lines='skip'
            )
        else:
            df = pd.read_csv(
                io.StringIO(raw),
                encoding='utf-8',
                on_bad_lines='skip'
            )
    else:
        df = pd.read_csv(dataset_path, encoding='utf-8', on_bad_lines='skip')

    df.columns = [col.strip() for col in df.columns]

    if 'label' not in df.columns:
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'})
        else:
            raise ValueError("Expected dataset to contain a 'label' column.")

    if 'text' not in df.columns and 'title' not in df.columns:
        raise ValueError("Expected dataset to contain 'text' or 'title' column.")

    if 'text' not in df.columns and 'title' not in df.columns:
        raise ValueError("Expected dataset to contain 'text' or 'title' column.")

    df = df.fillna('')

    if 'text' not in df.columns and 'title' in df.columns:
        df['text'] = df['title']

    if 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].str.cat(df['text'], sep=' ')

    return df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words='english',
        max_df=0.75,
        min_df=2,
        ngram_range=(1, 2),
        max_features=15000,
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        solver='liblinear',
        random_state=42,
        class_weight='balanced',
        max_iter=1000,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print('\n=== MODEL EVALUATION ===')
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred))


def predict_text(model, vectorizer, text: str) -> str:
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]


def main():
    parser = argparse.ArgumentParser(description='Fake News Detection with Logistic Regression')
    parser.add_argument(
        '--dataset',
        default=DATASET_PATH,
        help=f'Path to the CSV dataset file (default: {DATASET_PATH})'
    )
    args = parser.parse_args()

    try:
        df = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as error:
        print(f'Error: {error}')
        sys.exit(1)

    print('\nDataset preview:')
    print(df.head(5))
    print(f'\nDataset shape: {df.shape}')
    print('\nLabel distribution:')
    print(df['label'].value_counts())

    X = df['text']
    y = df['label']

    stratify_labels = None
    if len(np.unique(y)) > 1:
        label_counts = y.value_counts()
        if (label_counts >= 2).all():
            stratify_labels = y
        else:
            print('\nWarning: dataset has classes with fewer than 2 samples. Splitting without stratification.')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_labels,
    )

    vectorizer = build_vectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = build_model()
    model.fit(X_train_vectorized, y_train)

    evaluate_model(model, X_test_vectorized, y_test)

    print('\n=== CUSTOM NEWS TEST ===')
    while True:
        sample_text = input('\nEnter a news article to classify (or type QUIT to exit): ').strip()
        if sample_text.upper() in {'QUIT', 'EXIT'}:
            print('Exiting.')
            break

        if not sample_text:
            print('Please enter non-empty text.')
            continue

        prediction = predict_text(model, vectorizer, sample_text)
        label_text = 'FAKE' if str(prediction).strip().upper() == 'FAKE' else 'REAL'
        print(f'Prediction: {label_text}')


if __name__ == '__main__':
    main()
