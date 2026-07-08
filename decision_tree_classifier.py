import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# Data loading and preparation

def load_and_prepare_data(path="data_sets.xlsx", sheet="banking_customer_churn_predict"):
    data = pd.read_excel(path, sheet_name=sheet)
    print(data.head())
    print(data.describe())

    data = data.drop(columns=["CustomerId", "RowNumber", "Surname"])
    data = pd.get_dummies(data, columns=["Gender", "Geography"], drop_first=True)

    y = data["Exited"]
    X = data.drop(columns=["Exited"])
    feature_names = X.columns

    return X.to_numpy(), y.to_numpy(), feature_names

# Cross-validation

def eval_model_with_max_depths(X, y, cv, max_depths=range(1, 11)):
    """Runs CV for each depth and returns the raw scores per depth,
    so they can be reused later"""
    results = {}

    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
        results[depth] = scores

        print(
            f"Decision Tree (max_depth={depth}): "
            f"{scores.mean():.4f} (+/- {scores.std():.4f})"
        )

    return results

# Additional plots

def plot_decision_boundary(model, X, y, feature_names, feature_indices):
    X_selected = X[:, feature_indices]

    x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    scatter = plt.scatter(
        X_selected[:, 0],
        X_selected[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        edgecolor="k",
        s=20,
    )
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title("Decision Tree Decision Boundaries")
    plt.show()


def plot_accuracy_histogram(scores, title="Decision Tree Accuracy Distribution"):
    mean_score = scores.mean()

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=10, edgecolor="black")
    plt.axvline(mean_score, color="red", linestyle="--", label=f"Mean = {mean_score:.4f}")
    plt.legend()
    plt.title(title)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_confusion(y_test, y_pred, labels=("0", "1")):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

# Main

def main():
    X, y, feature_names = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    """Depth search: keep the raw CV scores per depth so we don't need
    to recompute cross_val_score again later for the best depth"""
    cv_results = eval_model_with_max_depths(X, y, cv)

    mean_accuracies = {depth: scores.mean() for depth, scores in cv_results.items()}
    best_max_depth = max(mean_accuracies, key=mean_accuracies.get)
    print(f"\nBest max_depth = {best_max_depth}")

    best_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    best_model.fit(X_train, y_train)

    # Tree visualization
    plt.figure(figsize=(24, 12))
    plot_tree(
        best_model,
        feature_names=feature_names,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.show()

    # Decision boundary
    feature_indices = [2, 6]
    model2 = DecisionTreeClassifier(max_depth=3, random_state=42)
    model2.fit(X_train[:, feature_indices], y_train)
    plot_decision_boundary(model2, X_train, y_train, feature_names, feature_indices)

    # Histogram accuracy (reuses scores computed in the depth search)
    plot_accuracy_histogram(cv_results[best_max_depth])

    # Model test
    y_pred = best_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    plot_confusion(y_test, y_pred)

if __name__ == "__main__":
    main()