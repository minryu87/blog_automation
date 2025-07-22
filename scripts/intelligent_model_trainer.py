import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/feature_calculate"
MASTER_DATA_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/data_processed/master_post_data.csv"
MODEL_OUTPUT_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/trained_models"
CHART_OUTPUT_PATH = "/Users/min/codes/medilawyer_sales/blog_automation/data/modeling/charts"

TARGETS = {
    'ctr': 'non_brand_average_ctr',
    'inflow': 'non_brand_inflow'
}

FEATURE_FILES = {
    'ctr': 'ctr_feature_value.csv',
    'inflow': 'inflow_feature_value.csv'
}

# --- Core Functions ---

def load_and_prepare_data(target_key: str):
    """Loads feature data and master data, merges them, and prepares for modeling."""
    print(f"\n--- 1. Loading and preparing data for '{target_key.upper()}' model ---")
    
    # Load data
    feature_df = pd.read_csv(os.path.join(DATA_PATH, FEATURE_FILES[target_key]))
    master_df = pd.read_csv(MASTER_DATA_PATH)

    # Define target and identifiers
    target_col = TARGETS[target_key]
    identifier_cols = ['post_identifier']
    
    # Merge data
    data = pd.merge(feature_df, master_df[identifier_cols + [target_col]], on='post_identifier')

    # Clean data
    data = data.dropna(subset=[target_col])
    data = data.select_dtypes(include=np.number) # Use only numeric features
    data = data.fillna(data.median()) # Impute missing values with median

    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    print(f"Data prepared. Shape: {X.shape}. Target: '{target_col}'")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains and evaluates a portfolio of models."""
    print("\n--- 2. Training and evaluating individual models ---")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LGBM": LGBMRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results.append({'model': name, 'r2_score': r2, 'mae': mae})
        print(f"  - {name}: R² = {r2:.4f}, MAE = {mae:.4f}")
        
    return pd.DataFrame(results)

def perform_feature_selection(estimator, X_train, y_train):
    """Performs Recursive Feature Elimination with Cross-Validation."""
    print(f"\n--- 3. Performing RFECV with {estimator.__class__.__name__} ---")
    
    selector = RFECV(estimator=estimator, step=1, cv=5, scoring='r2', n_jobs=-1)
    selector.fit(X_train, y_train)
    
    print(f"Optimal number of features found: {selector.n_features_}")
    selected_features = X_train.columns[selector.support_]
    print(f"Selected features: {list(selected_features)}")
    
    return selected_features

def train_stacking_ensemble(base_models, X_train, y_train):
    """Trains a stacking ensemble model."""
    print("\n--- 4. Training Stacking Ensemble Model ---")
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    # Meta-model to combine the predictions of the base models
    final_estimator = GradientBoostingRegressor(random_state=42)
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5
    )
    
    stacking_regressor.fit(X_train, y_train)
    return stacking_regressor

def plot_feature_importance(model, feature_names, target_key):
    """Plots and saves feature importance."""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model.__class__.__name__} does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=forest_importances, y=forest_importances.index)
    plt.title(f'Feature Importances for {target_key.upper()} Model')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    chart_file = os.path.join(CHART_OUTPUT_PATH, f'{target_key}_feature_importance.png')
    os.makedirs(CHART_OUTPUT_PATH, exist_ok=True)
    plt.savefig(chart_file)
    plt.close()
    print(f"Feature importance chart saved to {chart_file}")

# --- Main Orchestration ---
def intelligent_training_pipeline(target_key: str):
    """Runs the full intelligent training pipeline for a given target."""
    
    # 1. Data Preparation
    X_train, y_train, X_test, y_test, scaler = load_and_prepare_data(target_key)

    if X_train is None:
        print(f"'{target_key}' 모델 훈련을 위한 데이터 준비에 실패하여 건너뜁니다.")
        return

    # 2. Initial Model Evaluation
    initial_results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    print("\nInitial Model Performance:")
    print(initial_results)
    
    # 3. Feature Selection with Best Performing Model
    best_initial_model_name = initial_results.sort_values('r2_score', ascending=False).iloc[0]['model']
    print(f"\nBest initial model: {best_initial_model_name}")
    
    # Re-initialize the best model for RFECV
    models = {
        "Linear Regression": LinearRegression(), "Ridge": Ridge(random_state=42), "Lasso": Lasso(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42), "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LGBM": LGBMRegressor(random_state=42), "XGBoost": XGBRegressor(random_state=42)
    }
    best_estimator = models[best_initial_model_name]
    
    selected_features = perform_feature_selection(best_estimator, X_train, y_train)
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # 4. Train best model with selected features
    print(f"\nTraining {best_initial_model_name} with selected features...")
    best_model_optimized = models[best_initial_model_name]
    best_model_optimized.fit(X_train_selected, y_train)
    preds_optimized = best_model_optimized.predict(X_test_selected)
    r2_optimized = r2_score(y_test, preds_optimized)
    mae_optimized = mean_absolute_error(y_test, preds_optimized)
    print(f"Optimized {best_initial_model_name}: R² = {r2_optimized:.4f}, MAE = {mae_optimized:.4f}")

    # 5. Train Stacking Ensemble
    top_models = {name: models[name] for name in initial_results.sort_values('r2_score', ascending=False).head(3)['model']}
    stacking_model = train_stacking_ensemble(top_models, X_train_selected, y_train)
    stacking_preds = stacking_model.predict(X_test_selected)
    r2_stacking = r2_score(y_test, stacking_preds)
    mae_stacking = mean_absolute_error(y_test, stacking_preds)
    print(f"Stacking Ensemble: R² = {r2_stacking:.4f}, MAE = {mae_stacking:.4f}")

    # 6. Champion Model Selection and Saving
    champion_model = stacking_model if r2_stacking > r2_optimized else best_model_optimized
    champion_name = "Stacking Ensemble" if r2_stacking > r2_optimized else f"Optimized {best_initial_model_name}"
    
    print(f"\n--- Champion Model for {target_key.upper()} is: {champion_name} ---")

    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    model_filename = os.path.join(MODEL_OUTPUT_PATH, f'{target_key}_champion_model.joblib')
    joblib.dump(champion_model, model_filename)
    print(f"Champion model saved to {model_filename}")

    # 7. Feature Importance of the best single model (more interpretable)
    plot_feature_importance(best_model_optimized, selected_features, target_key)

if __name__ == "__main__":
    intelligent_training_pipeline('ctr')
    intelligent_training_pipeline('inflow')
    print("\n\n--- All modeling pipelines completed. ---") 