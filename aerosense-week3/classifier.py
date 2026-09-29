"""
classifier.py  --  Week 3, Task 2: Classification

Basic classifier on the synthetic dataset (rf_signature_dataset_v1.csv).
Classes: UAV, WiFi, Generic, Background  (simulated signal categories).

Models compared: Decision tree, Random forest, SVM, k-NN.
Output: accuracy + confusion matrix for each model.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV = sys.argv[1] if len(sys.argv) > 1 else "rf_signature_dataset_v1.csv"

FEATURES = ["frequency", "bandwidth", "power", "snr", "duration",
            "burst_interval", "spectral_flatness", "papr_db", "noise_floor_dbm"]
LABEL = "signal_type"
CLASSES = ["UAV", "WiFi", "Generic", "Background"]


def load_data(path=CSV):
    df = pd.read_csv(path)
    X = df[FEATURES].fillna(0.0)          # Background has no freq/bandwidth/interval
    y = df[LABEL]
    return X, y


def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main():
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    print(f"Samples: {len(X)}  (train {len(X_tr)}, test {len(X_te)})")
    print(f"Features: {', '.join(FEATURES)}\n")

    results = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        results[name] = acc
        print("=" * 60)
        print(f"{name}   accuracy = {acc:.3f}")
        print("Confusion matrix (rows = actual, columns = predicted)")
        cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=CLASSES),
                          index=CLASSES, columns=CLASSES)
        print(cm.to_string())
        print()
        print(classification_report(y_te, pred, labels=CLASSES, digits=3))

    print("=" * 60)
    print("Summary")
    for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {acc:.3f}")


if __name__ == "__main__":
    main()
