"""Task 2 - Classification: basic classifier on the Week 2 synthetic dataset (RF Signature Dataset v1).
Decision tree, random forest, SVM, k-NN."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from aerosense.engine import load_dataset
from aerosense.features import FEATURES

X, y = load_dataset()
print("Dataset:", X.shape, dict(zip(*np.unique(y, return_counts=True))), "\nFeatures:", FEATURES)
models = {
    "Decision tree": DecisionTreeClassifier(max_depth=8, random_state=0),
    "Random forest": RandomForestClassifier(n_estimators=200, random_state=0),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
    "k-NN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
}
print("\n5-fold cross-validated accuracy:")
for n, m in models.items():
    s = cross_val_score(m, X, y, cv=5)
    print(f"  {n:<15} {s.mean():.4f} ± {s.std():.4f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
rf = models["Random forest"].fit(Xtr, ytr)
pred = rf.predict(Xte)
print("\nRandom forest hold-out report:\n", classification_report(yte, pred, digits=3))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=rf.classes_), display_labels=rf.classes_).plot(
    ax=ax[0], colorbar=False, cmap="Blues")
ax[0].set_title("Confusion matrix - Random forest")
o = np.argsort(rf.feature_importances_)
ax[1].barh(np.array(FEATURES)[o], rf.feature_importances_[o]); ax[1].set_title("Feature importance")
plt.tight_layout(); plt.savefig("results/task2_classification.png", dpi=150)
