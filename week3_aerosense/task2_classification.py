"""
TASK 2 - Classification: basic classifier on the Week 2 synthetic dataset.
Compares Decision tree, Random forest, SVM and k-NN; reports accuracy, confusion matrices and
feature importance; saves the best model for the AeroSense engine.

Run:  python task2_classification.py
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from aerosense import RESULTS, MODELS
from aerosense.classifier import load_dataset, FEATURES, CLASSES

RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
X, y = load_dataset()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

models = {
    "Decision tree": DecisionTreeClassifier(max_depth=8, random_state=0),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
    "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows, cms = [], {}
for name, m in models.items():
    cvs = cross_val_score(m, Xtr, ytr, cv=cv)
    m.fit(Xtr, ytr); pred = m.predict(Xte)
    rows.append(dict(model=name, cv_accuracy=cvs.mean(), cv_std=cvs.std(), test_accuracy=(pred == yte).mean()))
    cms[name] = confusion_matrix(yte, pred, labels=CLASSES)
res = pd.DataFrame(rows).round(4); res.to_csv(RESULTS / "task2_model_comparison.csv", index=False)
print(res.to_string(index=False))

best = res.sort_values(["cv_accuracy", "test_accuracy"], ascending=False).iloc[0]["model"]
best_model = models[best]
print("\nBest model (5-fold CV on training split):", best)
print(classification_report(yte, best_model.predict(Xte), digits=3))

# confusion matrices (all four models)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (name, cm) in zip(axs, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"{name}\naccuracy {res.set_index('model').loc[name, 'test_accuracy']:.3f}")
fig.suptitle("Task 2 - Confusion matrices (held-out 25% test set)"); fig.tight_layout()
fig.savefig(RESULTS / "task2_confusion_matrices.png", dpi=140)

# feature importance (random forest) = which machine-learning features matter
rf = models["Random forest"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.6)); imp.plot.barh(ax=ax, color="tab:blue")
ax.set(xlabel="Importance", title="Task 2 - Random-forest feature importance"); fig.tight_layout()
fig.savefig(RESULTS / "task2_feature_importance.png", dpi=140)
print("\nFeature importance:\n", imp.sort_values(ascending=False).round(3).to_string())

# final model for the engine: retrain best on ALL data
final = models[best].__class__(**models[best].get_params()) if best in ("Decision tree", "Random forest") else \
        make_pipeline(StandardScaler(), (SVC(probability=True, random_state=0) if best == "SVM" else KNeighborsClassifier(5)))
final.fit(X, y)
joblib.dump(dict(model=final, features=FEATURES, classes=list(final.classes_), name=best), MODELS / "aerosense_classifier.joblib")
json.dump(dict(best_model=best, features=FEATURES, table=res.to_dict("records"),
               importance=imp.sort_values(ascending=False).round(4).to_dict()),
          open(RESULTS / "task2_summary.json", "w"), indent=2)
