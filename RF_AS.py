import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tqdm
import shap

df = pd.read_csv("RF/All_SOR_LLM_new.csv")
feature_cols = df.columns[1:25].tolist() + [df.columns[37]]           # Financial
# feature_cols = df.columns[1:38].tolist()                              # Financial + GPT
# feature_cols = df.columns[1:38].tolist() + df.columns[47:].tolist()     # Financial + GPT + LLM

features = df[feature_cols]
results = df.iloc[:, 38:47]

predictions = pd.DataFrame(index=results.index, columns=results.columns, dtype=float)
all_shap_values = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for col in tqdm.tqdm(results.columns):
    col_predictions = np.zeros(len(results))
    col_shap_values = []
    
    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train = results.loc[train_index, col]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        col_predictions[test_index] = rf.predict(X_test)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        col_shap_values.append(np.abs(shap_values).mean(axis=0))
    
    predictions[col] = col_predictions
    all_shap_values.append(np.mean(col_shap_values, axis=0))

best_algorithm_indices = predictions.idxmax(axis=1)

selected_true_values = []
for idx, best_algo in enumerate(best_algorithm_indices):
    selected_true_values.append(results.loc[idx, best_algo])

average_true_value = np.mean(selected_true_values)

print(f"Selected Max Average Value: {average_true_value}")

mean_shap_values = np.mean(all_shap_values, axis=0)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Shapley_Importance': mean_shap_values
})
feature_importance = feature_importance.sort_values('Shapley_Importance', ascending=False)

print("Shapley Importance")
print(feature_importance.to_string(index=False))
