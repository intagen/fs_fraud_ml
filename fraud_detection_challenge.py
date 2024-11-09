import warnings
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    make_scorer,
    roc_curve  # Added roc_curve here
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)

path = kagglehub.dataset_download("goyaladi/fraud-detection-dataset")
print("Path to dataset files:", path)

data_path = f"{path}/Data/"

transactions_df = pd.read_csv(f"{data_path}Transaction Data/transaction_records.csv")
metadata_df = pd.read_csv(f"{data_path}Transaction Data/transaction_metadata.csv")
transactions = pd.merge(transactions_df, metadata_df, on="TransactionID", how="left")

fraud_indicators_df = pd.read_csv(f"{data_path}Fraudulent Patterns/fraud_indicators.csv")
suspicious_activity_df = pd.read_csv(f"{data_path}Fraudulent Patterns/suspicious_activity.csv")
transactions = pd.merge(transactions, fraud_indicators_df, on="TransactionID", how="left")
transactions = pd.merge(transactions, suspicious_activity_df, on="CustomerID", how="left")

merchant_data_df = pd.read_csv(f"{data_path}Merchant Information/merchant_data.csv")
transaction_category_labels_df = pd.read_csv(
    f"{data_path}Merchant Information/transaction_category_labels.csv"
)

print("Transactions dataframe columns:", transactions.columns.tolist())
print("Merchant data dataframe columns:", merchant_data_df.columns.tolist())
print(
    "Transaction category labels dataframe columns:",
    transaction_category_labels_df.columns.tolist(),
)

transactions = pd.merge(transactions, merchant_data_df, on="MerchantID", how="left")

amount_data_df = pd.read_csv(f"{data_path}Transaction Amounts/amount_data.csv")
anomaly_scores_df = pd.read_csv(f"{data_path}Transaction Amounts/anomaly_scores.csv")
transactions = pd.merge(transactions, amount_data_df, on="TransactionID", how="left")
transactions = pd.merge(transactions, anomaly_scores_df, on="TransactionID", how="left")

customer_data_df = pd.read_csv(f"{data_path}Customer Profiles/customer_data.csv")
account_activity_df = pd.read_csv(f"{data_path}Customer Profiles/account_activity.csv")
transactions = pd.merge(transactions, customer_data_df, on="CustomerID", how="left")
transactions = pd.merge(transactions, account_activity_df, on="CustomerID", how="left")

transactions["FraudIndicator"] = transactions["FraudIndicator"].fillna(0)
transactions["SuspiciousFlag"] = transactions["SuspiciousFlag"].fillna(0)
transactions["is_fraud"] = np.where(
    (transactions["FraudIndicator"] == 1) | (transactions["SuspiciousFlag"] == 1), 1, 0
)
transactions.drop(["FraudIndicator", "SuspiciousFlag"], axis=1, inplace=True)
transactions.drop_duplicates(inplace=True)

transactions["Timestamp"] = pd.to_datetime(transactions["Timestamp"])

if "AccountCreationDate" in transactions.columns:
    transactions["AccountCreationDate"] = pd.to_datetime(transactions["AccountCreationDate"])
else:
    transactions["AccountCreationDate"] = transactions["Timestamp"] - pd.to_timedelta(
        np.random.randint(30, 365, size=len(transactions)), unit="d"
    )

transactions.fillna(0, inplace=True)

transactions.sort_values(["CustomerID", "Timestamp"], inplace=True)

transactions["TimeSinceLastTransaction"] = transactions.groupby("CustomerID")["Timestamp"].diff().dt.total_seconds()
transactions["TimeSinceLastTransaction"].fillna(0, inplace=True)

transactions["TransactionCount"] = transactions.groupby("CustomerID")["TransactionID"].transform("count")
transactions["TotalTransactionAmount"] = transactions.groupby("CustomerID")["Amount"].transform("sum")
transactions["AvgTransactionAmount"] = transactions["TotalTransactionAmount"] / transactions["TransactionCount"]
transactions["AmountDeviation"] = transactions["Amount"] - transactions["AvgTransactionAmount"]

transactions["Hour"] = transactions["Timestamp"].dt.hour
transactions["Day"] = transactions["Timestamp"].dt.day
transactions["Month"] = transactions["Timestamp"].dt.month
transactions["Weekday"] = transactions["Timestamp"].dt.weekday

transactions["IsWeekend"] = transactions["Weekday"].apply(lambda x: 1 if x >= 5 else 0)
transactions["AccountAgeDays"] = (transactions["Timestamp"] - transactions["AccountCreationDate"]).dt.days

transactions.drop(["Timestamp", "AccountCreationDate"], axis=1, inplace=True)

categorical_cols = [
    "MerchantCategory",
    "TransactionType",
    "CustomerSegment",
    "Gender",
    "TransactionCategoryLabel",
]

for col in categorical_cols:
    if col in transactions.columns:
        transactions[col] = transactions[col].astype("category").cat.codes

transactions.reset_index(drop=True, inplace=True)

target = "is_fraud"
features = transactions.columns.drop([target])

X = transactions[features]
y = transactions[target]

non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Non-numeric columns:", non_numeric_cols)

label_enc_cols = [col for col in non_numeric_cols if col in ["MerchantName", "Location"]]
X.drop([col for col in non_numeric_cols if col not in ["MerchantName", "Location"]], axis=1, inplace=True)

le = LabelEncoder()
for col in label_enc_cols:
    if col in X.columns:
        X[col] = le.fit_transform(X[col].astype(str))

one_hot_cols = ["CustomerSegment", "Gender"]
X = pd.get_dummies(X, columns=[col for col in one_hot_cols if col in X.columns], drop_first=True)

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_full.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train_full.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_full, y_train_full)
print(f"Resampled training set shape: {X_resampled.shape}")

results = {}

def evaluate_model(model, model_name):
    model.fit(X_resampled, y_resampled)
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\n{model_name} - Best threshold based on f1 score: {best_threshold:.4f}")
    y_pred = (y_probs >= best_threshold).astype(int)
    print(f"\n{model_name} - Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"{model_name} - Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    roc_auc = roc_auc_score(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)
    print(f"{model_name} - ROC AUC score: {roc_auc:.4f}")
    print(f"{model_name} - Average precision score: {avg_precision:.4f}")
    
    results[model_name] = {
        "model": model,
        "best_threshold": best_threshold,
        "classification_report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
        "confusion_matrix": cm,
        "roc_auc_score": roc_auc,
        "average_precision_score": avg_precision,
    }
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall_vals, precision_vals, label=f"Average Precision = {avg_precision:.2f}")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

print("\nTraining XGBoost Classifier...")

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    verbosity=0,
    use_label_encoder=False,
    random_state=42,
)

def xgb_objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }
    xgb_clf.set_params(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {"f1": make_scorer(f1_score, pos_label=1)}
    scores = cross_validate(
        xgb_clf,
        X_resampled,
        y_resampled,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return scores["test_f1"].mean()

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=20)

xgb_best_params = xgb_study.best_params
print("\nxgboost best hyperparameters:")
print(xgb_best_params)

xgb_clf.set_params(**xgb_best_params)
scale_pos_weight = y_resampled.value_counts()[0] / y_resampled.value_counts()[1]
xgb_clf.set_params(scale_pos_weight=scale_pos_weight)

evaluate_model(xgb_clf, "XGBoost")

print("\nTraining Random Forest Classifier...")

rf_clf = RandomForestClassifier(random_state=42)

def rf_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),
        "n_jobs": -1,
    }
    rf_clf.set_params(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {"f1": make_scorer(f1_score, pos_label=1)}
    scores = cross_validate(
        rf_clf,
        X_resampled,
        y_resampled,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return scores["test_f1"].mean()

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=20)

rf_best_params = rf_study.best_params
print("\nrandom forest best hyperparameters:")
print(rf_best_params)

rf_clf.set_params(**rf_best_params)
evaluate_model(rf_clf, "Random Forest")

print("\nTraining LightGBM Classifier...")

lgb_clf = lgb.LGBMClassifier(random_state=42, verbosity=-1)

def lgb_objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 42,
        "verbosity": -1,
    }
    lgb_clf.set_params(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {"f1": make_scorer(f1_score, pos_label=1)}
    scores = cross_validate(
        lgb_clf,
        X_resampled,
        y_resampled,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return scores["test_f1"].mean()

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=20)

lgb_best_params = lgb_study.best_params
print("\nlightgbm best hyperparameters:")
print(lgb_best_params)

lgb_clf.set_params(**lgb_best_params)
evaluate_model(lgb_clf, "LightGBM")

print("\nTraining CatBoost Classifier...")

cb_clf = CatBoostClassifier(verbose=0, random_state=42)

def cb_objective(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }
    cb_clf.set_params(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {"f1": make_scorer(f1_score, pos_label=1)}
    scores = cross_validate(
        cb_clf,
        X_resampled,
        y_resampled,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    return scores["test_f1"].mean()

cb_study = optuna.create_study(direction="maximize")
cb_study.optimize(cb_objective, n_trials=20)

cb_best_params = cb_study.best_params
print("\ncatboost best hyperparameters:")
print(cb_best_params)

cb_clf.set_params(**cb_best_params)
evaluate_model(cb_clf, "CatBoost")

comparison_list = []

for model_name, result in results.items():
    clf_report = result["classification_report"]
    precision = clf_report["1"]["precision"]
    recall = clf_report["1"]["recall"]
    f1 = clf_report["1"]["f1-score"]
    roc_auc = result["roc_auc_score"]
    avg_precision = result["average_precision_score"]
    comparison_list.append(
        {
            "Model": model_name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC AUC": roc_auc,
            "Average Precision": avg_precision,
        }
    )

comparison_df = pd.DataFrame(comparison_list)
print("\nmodel comparison:")
print(comparison_df)

plt.figure(figsize=(12, 8))
metrics = ["Precision", "Recall", "F1-Score", "ROC AUC", "Average Precision"]
comparison_melted = comparison_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

sns.barplot(x="Model", y="Score", hue="Metric", data=comparison_melted)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

best_model_row = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
best_model = results[best_model_name]["model"]
best_threshold = results[best_model_name]["best_threshold"]

joblib.dump(best_model, f"{best_model_name}_fraud_detection_model.pkl")
print(f"\nbest model saved: {best_model_name}_fraud_detection_model.pkl")

new_data = X_test.iloc[:5]
new_preds = best_model.predict_proba(new_data)[:, 1]
new_preds_binary = (new_preds >= best_threshold).astype(int)

new_transactions = new_data.copy()
new_transactions["fraud_probability"] = new_preds
new_transactions["is_fraud"] = new_preds_binary

print(f"\nnew data predictions using {best_model_name}:")
print(new_transactions[["fraud_probability", "is_fraud"]])
