import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.font_manager as fm

def build_seo_prediction_model():
    """
    Builds an ML model to predict non-branded search inflow and derives an SEO scoring algorithm.
    1. Loads and cleans data, removing brand-related keywords.
    2. Defines the target variable as the sum of non-branded search inflow.
    3. Trains a LightGBM regression model.
    4. Evaluates the model's performance.
    5. Extracts and saves feature importances, which serve as the SEO scoring criteria.
    """
    # --- 1. Setup ---
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    nside_path = os.path.join(base_dir, 'data', 'data_processed', 'post_analysis_results.csv')
    query_inflow_path = os.path.join(base_dir, 'data', 'data_input', 'post-searchQuery.csv')
    output_dir = os.path.join(base_dir, 'data', 'data_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Font setup for Korean
    try:
        if os.name == 'posix':
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
            if os.path.exists(font_path):
                plt.rc('font', family='AppleGothic')
            else:
                print("AppleGothic font not found.")
        elif os.name == 'nt':
            font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print("Font setup for plots complete.")
    except Exception as e:
        print(f"Font setup failed: {e}")

    # --- 2. Data Loading and Cleaning ---
    try:
        df_nside = pd.read_csv(nside_path)
        df_query_inflow = pd.read_csv(query_inflow_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Filter out brand keywords
    brand_keyword = '내이튼'
    df_non_brand = df_query_inflow[~df_query_inflow['searchQuery'].str.contains(brand_keyword, na=False)].copy()
    
    # Calculate the new target variable: sum of non-branded inflow per post
    df_target = df_non_brand.groupby('postId')['searchInflow'].sum().reset_index()
    df_target.rename(columns={'searchInflow': 'non_brand_inflow'}, inplace=True)

    # --- 3. Feature Engineering and Final Dataset Prep ---
    
    # Merge features (n-side) with the new target variable
    df_model_data = pd.merge(df_nside, df_target, on='postId', how='inner')

    # Convert posting_score to a numeric feature
    def score_to_numeric(score):
        if pd.isna(score): return 0
        score = str(score)
        if '준최' in score: return int(score.split('-')[0].replace('준최', ''))
        if '저품질' in score: return -1 * int(score.replace('저품질', ''))
        return 0
    df_model_data['posting_score_numeric'] = df_model_data['posting_score'].apply(score_to_numeric)

    # Define features (X) and target (y)
    features = [
        'posting_score_numeric', 'exposure_score', 'score_가독성', 'score_주제', 'score_콘텐츠',
        'score_키워드', 'score_형태소', 'score_후기/정보', 'content_with_space', 'word_count',
        'valid_images', 'invalid_images'
    ]
    target = 'non_brand_inflow'

    X = df_model_data[features].fillna(0) # Fill any remaining NaNs in features
    y = df_model_data[target]

    # --- 4. Model Training and Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining LightGBM model...")
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lgbm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("R-squared indicates the proportion of the variance in the dependent variable that is predictable from the independent variable(s). Higher is better.")


    # --- 5. Feature Importance (Scoring Criteria) ---
    df_importance = pd.DataFrame({
        'feature': features,
        'importance': lgbm.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- (Result) SEO Feature Importance (Scoring Criteria) ---")
    print(df_importance)
    
    # Save feature importances to CSV
    importance_path = os.path.join(output_dir, 'seo_feature_importances.csv')
    df_importance.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"\nFeature importances saved to {importance_path}")

    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=df_importance)
    plt.title('SEO Feature Importance for Predicting Non-Brand Inflow', fontsize=16)
    plt.tight_layout()
    
    importance_plot_path = os.path.join(output_dir, 'seo_feature_importance_plot.png')
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"Feature importance plot saved to {importance_plot_path}")

if __name__ == "__main__":
    build_seo_prediction_model() 