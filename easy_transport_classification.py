# ============================================================
# Easy Transport – Student Interest Classification
# Target: Interested_In_EasyTransport_App (Yes / No / Maybe)
# Models : Decision Tree | Random Forest | Gradient Boosting
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

# ── 2. Load Dataset ──────────────────────────────────────────
CSV_PATH = "easy_transport_survey.csv"          # <-- update path if needed
TARGET   = "Interested_In_EasyTransport_App"

df = pd.read_csv(CSV_PATH)

# ── 3. Basic Data Checks ─────────────────────────────────────
print("=" * 60)
print("SHAPE:", df.shape)
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nDATA TYPES:")
print(df.dtypes)
print("\nMISSING VALUES:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDESCRIPTIVE STATISTICS (numeric):")
print(df.describe())
print("\nTARGET DISTRIBUTION:")
print(df[TARGET].value_counts())

# ── 4. Preprocessing ─────────────────────────────────────────
# Separate target
y_raw = df[TARGET].copy()
X_raw = df.drop(columns=[TARGET])

# -- Identify column types
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()

# -- Impute missing values
#    Numeric  → median  (robust to outliers)
#    Categorical → most frequent
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_num = pd.DataFrame(
    num_imputer.fit_transform(X_raw[num_cols]),
    columns=num_cols, index=X_raw.index
)
X_cat_raw = pd.DataFrame(
    cat_imputer.fit_transform(X_raw[cat_cols]),
    columns=cat_cols, index=X_raw.index
)

# -- One-hot encode nominal categorical columns
X_cat_enc = pd.get_dummies(X_cat_raw, drop_first=False)

# -- Combine
X = pd.concat([X_num, X_cat_enc], axis=1)
X.columns = X.columns.astype(str)          # ensure string column names

# -- Encode target label
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw.astype(str))
class_names = le_target.classes_
print("\nTarget classes:", class_names)

# ── 5. Train / Test Split (80/20, stratified) ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 6. Train Three Models ────────────────────────────────────
models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=8),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                       max_depth=4, random_state=42),
}

results   = {}
fitted    = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    results[name] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc}
    fitted[name]  = model

# ── 7. Print Accuracy Table ──────────────────────────────────
print("\n" + "=" * 50)
print(f"{'Model':<22} {'Train Acc':>12} {'Test Acc':>12}")
print("-" * 50)
for name, acc in results.items():
    print(f"{name:<22} {acc['Train Accuracy']:>11.4f} {acc['Test Accuracy']:>11.4f}")
print("=" * 50)

# ── 8. Confusion Matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices – Easy Transport Interest Prediction", fontsize=14, fontweight="bold")

for ax, (name, model) in zip(axes, fitted.items()):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("Saved: confusion_matrices.png")

# ── 9. Feature Importance (Top 20 per model) ─────────────────
feat_names = X.columns.tolist()
n_top      = min(20, len(feat_names))

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle("Top Feature Importances – Easy Transport Classification",
             fontsize=14, fontweight="bold")

palette = {"Decision Tree": "#4C72B0",
           "Random Forest": "#55A868",
           "Gradient Boosting": "#C44E52"}

for ax, (name, model) in zip(axes, fitted.items()):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:n_top]
    top_feats   = [feat_names[i] for i in indices]
    top_vals    = importances[indices]

    sns.barplot(x=top_vals, y=top_feats, ax=ax, color=palette[name])
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()
print("Saved: feature_importances.png")

print("\nDone. Check confusion_matrices.png and feature_importances.png.")
