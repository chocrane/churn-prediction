"""Churn prediction with gradient boosting."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

def train(X, y) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42)
    clf.fit(X, y)
    return clf

def evaluate(clf, X_test, y_test) -> dict:
    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)[:, 1]
    rep  = classification_report(y_test, pred, output_dict=True)
    rep["roc_auc"] = roc_auc_score(y_test, prob)
    return rep

def save(clf, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(clf))

def load(path: Path) -> GradientBoostingClassifier:
    return pickle.loads(path.read_bytes())
