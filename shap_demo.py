# pip install shap scikit-learn matplotlib numpy pandas

import shap
import time  # <-- NEW: Track computation time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Load data with progress print ---
print("Step 1/5: Loading data...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# Use smaller subset for demo (2% instead of 20%)
print("Step 2/5: Preparing data...")
_, X_small, _, y_small = train_test_split(X, y, test_size=0.02, random_state=42)

# --- Train model ---
print("Step 3/5: Training model...")
start_time = time.time()
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_small, y_small)
print(f"Model trained in {time.time()-start_time:.1f} seconds")

# --- SHAP calculations ---
print("Step 4/5: Calculating SHAP values (this may take 1-2 minutes)...")
explainer = shap.TreeExplainer(model)
start_time = time.time()
shap_values = explainer.shap_values(X_small)  # Use small subset
print(f"SHAP values calculated in {time.time()-start_time:.1f} seconds")

# --- Save outputs ---
print("Step 5/5: Generating plots...")

# 1. Feature importance
plt.figure(figsize=(8,5))
shap.summary_plot(shap_values, X_small, plot_type="bar", show=False)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# 2. Summary plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_small, show=False)
plt.title("Feature Impact Analysis")
plt.savefig("shap_summary.png")
plt.close()

# 3. Individual explanation (first sample)
shap.save_html("force_plot.html", 
              shap.force_plot(explainer.expected_value,
                             shap_values[0,:],
                             X_small.iloc[0,:],
                             feature_names=california.feature_names))

print("""
SUCCESS! Created 3 files in your current directory:
1. feature_importance.png - Global feature rankings
2. shap_summary.png       - Detailed impact visualization
3. force_plot.html        - Individual prediction explanation

Open the HTML file in a web browser to see interactive visualization
""")