"""
Quick hyperparameter tuning + stacking ensemble script.
Saves tuned models and vectorizers in models/ and prints evaluation results.
Designed to run reasonably fast (limited n_iter and small CV).
"""
import os
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = "data/news.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found: {DATA_PATH}\n"
        f"Please run: python src/dataset_builder.py\n"
        f"Or place Fake.csv and True.csv in data/ folder first."
    )

print("🔎 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df['label'] = df['label'].astype(int)

X = df['text'].values
y = df['label'].values

# split into train/val/test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=RANDOM_STATE)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Common TF-IDF settings
tfidf = TfidfVectorizer(stop_words='english')

# 1) TF-IDF + LogisticRegression tuning
print("\n1) Tuning TF-IDF + LogisticRegression...")
pipeline_tfidf = Pipeline([
    ('tfidf', tfidf),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

param_dist_tfidf = {
    'tfidf__max_features': [3000, 5000, 8000],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__class_weight': [None, 'balanced']
}

search_tfidf = RandomizedSearchCV(pipeline_tfidf, param_dist_tfidf, n_iter=8, cv=3, n_jobs=1, random_state=RANDOM_STATE, verbose=1)
search_tfidf.fit(X_train, y_train)
print("Best TF-IDF params:", search_tfidf.best_params_)

# validate
tfidf_val_pred = search_tfidf.predict(X_val)
acc_tfidf = accuracy_score(y_val, tfidf_val_pred)
print("TF-IDF val accuracy:", acc_tfidf)

# save
joblib.dump(search_tfidf.best_estimator_, os.path.join(MODEL_DIR, 'tfidf_pipeline.pkl'))
# also save model & vectorizer separately for compatibility
best_tfidf = search_tfidf.best_estimator_.named_steps['clf']
best_vectorizer = search_tfidf.best_estimator_.named_steps['tfidf']
joblib.dump(best_tfidf, os.path.join(MODEL_DIR, 'tfidf_model.pkl'))
joblib.dump(best_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

# 2) Random Forest tuning
print("\n2) Tuning Random Forest...")
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))
])
param_dist_rf = {
    'tfidf__max_features': [3000, 5000, 8000],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 50, 100],
    'clf__class_weight': [None, 'balanced_subsample']
}
search_rf = RandomizedSearchCV(pipeline_rf, param_dist_rf, n_iter=8, cv=3, n_jobs=1, random_state=RANDOM_STATE, verbose=1)
search_rf.fit(X_train, y_train)
print("Best RF params:", search_rf.best_params_)
rf_val_pred = search_rf.predict(X_val)
acc_rf = accuracy_score(y_val, rf_val_pred)
print("RF val accuracy:", acc_rf)

# save
joblib.dump(search_rf.best_estimator_, os.path.join(MODEL_DIR, 'rf_pipeline.pkl'))
best_rf = search_rf.best_estimator_.named_steps['clf']
best_rf_vec = search_rf.best_estimator_.named_steps['tfidf']
joblib.dump(best_rf, os.path.join(MODEL_DIR, 'rf_model.pkl'))
joblib.dump(best_rf_vec, os.path.join(MODEL_DIR, 'rf_vectorizer.pkl'))

# 3) XGBoost tuning
print("\n3) Tuning XGBoost...")
pipeline_xgb = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1))
])
param_dist_xgb = {
    'tfidf__max_features': [3000,5000,8000],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'clf__n_estimators': [100,200],
    'clf__max_depth': [4,6,8],
    'clf__learning_rate': [0.05, 0.08, 0.1],
    'clf__subsample': [0.7,0.8,1.0]
}
search_xgb = RandomizedSearchCV(pipeline_xgb, param_dist_xgb, n_iter=8, cv=3, n_jobs=1, random_state=RANDOM_STATE, verbose=1)
search_xgb.fit(X_train, y_train)
print("Best XGB params:", search_xgb.best_params_)
xgb_val_pred = search_xgb.predict(X_val)
acc_xgb = accuracy_score(y_val, xgb_val_pred)
print("XGBoost val accuracy:", acc_xgb)

# save
joblib.dump(search_xgb.best_estimator_, os.path.join(MODEL_DIR, 'xgb_pipeline.pkl'))
best_xgb = search_xgb.best_estimator_.named_steps['clf']
best_xgb_vec = search_xgb.best_estimator_.named_steps['tfidf']
joblib.dump(best_xgb, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
joblib.dump(best_xgb_vec, os.path.join(MODEL_DIR, 'xgb_vectorizer.pkl'))

# 4) Build stacking ensemble using pipelines' predict_proba on val set
print("\n4) Building stacking ensemble...")
# prepare base estimators: use already best pipelines
estimators = [
    ('tfidf', search_tfidf.best_estimator_),
    ('rf', search_rf.best_estimator_),
    ('xgb', search_xgb.best_estimator_)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=1,
    passthrough=False
)

print("Fitting stacking classifier on train+val...")
# fit on combined train+val for best performance
stack.fit(X_trainval, y_trainval)

# evaluate on test
stack_pred = stack.predict(X_test)
acc_stack = accuracy_score(y_test, stack_pred)
print("Stacking ensemble test accuracy:", acc_stack)

# save stacking model
joblib.dump(stack, os.path.join(MODEL_DIR, 'stacking_ensemble.pkl'))

# Print summary
print('\n✅ Summary of validation accuracies:')
print(f'TF-IDF val acc: {acc_tfidf:.4f}')
print(f'RF val acc: {acc_rf:.4f}')
print(f'XGB val acc: {acc_xgb:.4f}')
print(f'Stack test acc: {acc_stack:.4f}')

print('\nModels saved to models/ folder')
