from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

ROOT = Path(__file__).resolve().parent
NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
FEATURES = ["Type"] + NUMERIC


def preprocessing():
    # Each model gets its own preprocessing fitted only on its training rows.
    return ColumnTransformer([
        ("measurements", StandardScaler(), NUMERIC),
        ("product_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
    ])


def main(show=True):
    data = pd.read_csv(ROOT / "ai4i2020.csv")
    X, y = data[FEATURES], data["Machine failure"]
    # IDs and failure-type flags are excluded: flags would reveal the answer.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    output = ROOT / "results"
    charts = output / "combined_charts"
    charts.mkdir(parents=True, exist_ok=True)
    print(f"Records: {len(data):,} | Train: {len(X_train):,} (80%) | Test: {len(X_test):,} (20%)")

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Support Vector Machine": SVC(kernel="rbf", class_weight="balanced"),
    }
    models = {name: make_pipeline(preprocessing(), model) for name, model in algorithms.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Select with training CV F1, never with the held-out test results.
    cv_scores = {name: cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()
                 for name, model in models.items()}
    selected = max(cv_scores, key=cv_scores.get)
    rows, reports = [], []
    predictions = X_test.copy()
    predictions["actual_failure"] = y_test
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append({"algorithm": name, "training_cv_f1": cv_scores[name],
                     "test_accuracy": accuracy_score(y_test, predicted),
                     "test_precision": precision_score(y_test, predicted, zero_division=0),
                     "test_recall": recall_score(y_test, predicted, zero_division=0),
                     "test_f1": f1_score(y_test, predicted, zero_division=0)})
        matrix = pd.DataFrame(confusion_matrix(y_test, predicted, labels=[0, 1]),
                              index=["actual_no", "actual_failure"],
                              columns=["predicted_no", "predicted_failure"])
        reports.append(f"{name}\n{classification_report(y_test, predicted, zero_division=0)}\n{matrix}\n")
    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by training CV F1: {selected} (ties: first listed)")
    print(f"Always predicting no failure gives {1-y_test.mean():.2%} test accuracy.")
    print("Compare failure precision, recall and F1, not just accuracy.")
    example = pd.DataFrame([["M", 298.1, 308.6, 1551, 42.8, 0]], columns=FEATURES)
    print("Example prediction (0=no failure, 1=failure):", models[selected].predict(example)[0])
    comparison.to_csv(output / "model_comparison.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index_label="original_row_index")
    data.loc[X_train.index].to_csv(output / "training_data.csv", index=False)
    data.loc[X_test.index].to_csv(output / "testing_data.csv", index=False)
    (output / "classification_reports.txt").write_text("\n\n".join(reports), encoding="utf-8")

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(charts / filename, dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)

    # CHART 1: grouped bar chart of all five models' fresh test results.
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.set_index("algorithm")[["test_accuracy", "test_precision", "test_recall", "test_f1"]].plot.bar(ax=ax)
    ax.set(title="Five algorithms: test performance", ylabel="Score (0 to 1)", xlabel="Algorithm", ylim=(0, 1.1))
    ax.tick_params(axis="x", rotation=20)
    save(fig, "01_bar_model_comparison.png")

    # CHART 2: scatter plot. Draw failures last to keep them visible.
    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color, label in [(0, "steelblue", "No failure"), (1, "tomato", "Failure")]:
        subset = data[data["Machine failure"] == group]
        ax.scatter(subset["Rotational speed [rpm]"], subset["Torque [Nm]"],
                   color=color, label=label, s=12, alpha=0.4 if group == 0 else 0.8)
    ax.set(title="Speed versus torque", xlabel="Rotational speed [rpm]", ylabel="Torque [Nm]")
    ax.legend()
    save(fig, "02_scatter_speed_torque.png")

    # CHART 3: histogram panels for the five numeric inputs.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.flat, NUMERIC):
        ax.hist(data[column], bins=30, color="steelblue", edgecolor="white")
        ax.set(xlabel=column, ylabel="Count", title=column)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Measurement distributions")
    save(fig, "03_histograms.png")

    # CHART 4: box plot showing torque distributions by outcome.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([data.loc[data["Machine failure"] == group, "Torque [Nm]"] for group in [0, 1]],
               tick_labels=["No failure", "Failure"])
    ax.set(title="Torque grouped by machine outcome", ylabel="Torque [Nm]")
    save(fig, "04_boxplot_torque.png")

    # CHART 5: pie chart of class proportions with counts.
    counts = data["Machine failure"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(counts, labels=[f"No failure (n={counts[0]:,})", f"Failure (n={counts[1]:,})"],
           autopct="%.2f%%", colors=["steelblue", "tomato"], startangle=90)
    ax.set_title("Machine failure proportions")
    save(fig, "05_pie_failure_proportions.png")

    # CHART 6: line chart over record order, not elapsed time.
    sample = data.head(200)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sample["UDI"], sample["Air temperature [K]"], label="Air temperature")
    ax.plot(sample["UDI"], sample["Process temperature [K]"], label="Process temperature")
    ax.set(title="Temperature across the first 200 records", xlabel="Record ID (not time)", ylabel="Temperature [K]")
    ax.legend()
    save(fig, "06_line_temperatures.png")

    # CHART 7: correlation heatmap. Correlation does not establish causation.
    corr = data[NUMERIC + ["Machine failure"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    heat = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for row in range(len(corr)):
        for col in range(len(corr)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[row, col]) > 0.6 else "black")
    fig.colorbar(heat, ax=ax, label="Pearson correlation")
    ax.set_title("Measurement and failure correlations")
    save(fig, "07_correlation_heatmap.png")

    print(f"\nSeven chart files saved to: {charts}")
    print("Full-dataset charts are descriptive; use training data alone for model tuning.")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
