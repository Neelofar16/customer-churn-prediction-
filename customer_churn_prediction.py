# ✅ Step 1: Upload Dataset
from google.colab import files
uploaded = files.upload()

# ✅ Step 2: Load Data
import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()

# ✅ Step 3: Clean Data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop(['customerID'], axis=1, inplace=True)

# ✅ Step 4: Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include='object'):
    if col != 'Churn':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# ✅ Step 5: Train-Test Split
from sklearn.model_selection import train_test_split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# ✅ Step 7: Evaluate Models
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
models = {'Logistic Regression': log_model, 'Random Forest': rf_model, 'XGBoost': xgb_model}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n📌 {name} Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")

# ✅ Step 8: Plot ROC Curves
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_probs):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('📉 ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# ✅ Step 9: Feature Importance (XGBoost)
import seaborn as sns
importances = xgb_model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("💡 Top 10 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ✅ Step 10: Export Predictions for Power BI
output_df = X_test.copy()
output_df['Actual'] = y_test.values
output_df['Predicted'] = xgb_model.predict(X_test)
output_df['Churn_Probability'] = xgb_model.predict_proba(X_test)[:, 1]
output_df.to_csv('churn_predictions.csv', index=False)
print("✅ Saved Power BI-ready output to churn_predictions.csv")

# ✅ Step 11: Download Output (Colab Only)
files.download('churn_predictions.csv')
