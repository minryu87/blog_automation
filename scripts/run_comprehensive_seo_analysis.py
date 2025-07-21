import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def setup_environment(base_dir):
    """Sets up file paths and fonts for plotting."""
    paths = {
        'nside': os.path.join(base_dir, 'data', 'data_processed', 'post_analysis_results.csv'),
        'query_inflow': os.path.join(base_dir, 'data', 'data_input', 'post-searchQuery.csv'),
        'search_volume': os.path.join(base_dir, 'data', 'data_processed', 'keyword_search_volume_transform.csv'),
        'output_dir': os.path.join(base_dir, 'data', 'data_analysis')
    }
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    try:
        if os.name == 'posix':
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
            if os.path.exists(font_path): plt.rc('font', family='AppleGothic')
        elif os.name == 'nt':
            font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
            plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print("Font setup for plots complete.")
    except Exception as e:
        print(f"Font setup failed: {e}")
        
    return paths

def load_and_prepare_data(paths):
    """Loads, cleans, and engineers features for the analysis."""
    print("\nStep 1: Loading and Preparing Data...")
    df_nside = pd.read_csv(paths['nside'])
    df_query_inflow = pd.read_csv(paths['query_inflow'])
    df_search_volume = pd.read_csv(paths['search_volume'])

    # --- Create non-brand targets ---
    brand_keyword = '내이튼'
    df_non_brand = df_query_inflow[~df_query_inflow['searchQuery'].str.contains(brand_keyword, na=False)].copy()
    
    # Target 1: Non-brand inflow
    df_target_inflow = df_non_brand.groupby('postId')['searchInflow'].sum().reset_index()
    df_target_inflow.rename(columns={'searchInflow': 'non_brand_inflow'}, inplace=True)
    
    # Target 2: Non-brand average C
    df_non_brand['date'] = pd.to_datetime(df_non_brand['startDate']).dt.strftime('%Y-%m')
    df_search_volume['total_volume'] = df_search_volume['pc'] + df_search_volume['mobile']
    df_detailed = pd.merge(df_non_brand, df_search_volume, on=['searchQuery', 'date'], how='left')
    df_detailed['C'] = (df_detailed['searchInflow'] / df_detailed['total_volume']).fillna(0)
    df_detailed.replace([np.inf, -np.inf], 0, inplace=True)
    
    df_target_c = df_detailed.groupby('postId')['C'].mean().reset_index()
    df_target_c.rename(columns={'C': 'non_brand_average_C'}, inplace=True)

    # --- Create master dataset ---
    df_master = pd.merge(df_nside, df_target_inflow, on='postId', how='inner')
    df_master = pd.merge(df_master, df_target_c, on='postId', how='left').fillna(0)

    # Feature engineering for posting_score
    def score_to_numeric(score):
        score = str(score)
        if '준최' in score: return int(score.split('-')[0].replace('준최', ''))
        if '저품질' in score: return -1 * int(score.replace('저품질', ''))
        return 0
    df_master['posting_score_numeric'] = df_master['posting_score'].apply(score_to_numeric)
    print("Data preparation complete.")
    return df_master

def run_regression_analysis(df, features, target, output_dir):
    """Runs regression with multiple models and saves results."""
    print(f"\n--- Approach 1: Regression Analysis for '{target}' ---")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    results = []
    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results.append({'model': name, 'target': target, 'r2_score': r2})
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    df_results = pd.DataFrame(results)
    print("Model Performance Comparison:")
    print(df_results)

    # Feature importance from the best model
    if hasattr(best_model, 'feature_importances_'):
        df_importance = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importances from Best Model ({type(best_model).__name__}):")
        print(df_importance)
        
        # Save results
        path = os.path.join(output_dir, f'feature_importance_{target}.csv')
        df_importance.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Importance data saved to {path}")

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=df_importance)
        plt.title(f"Feature Importance for Predicting '{target}'", fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'feature_importance_{target}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Importance plot saved to {plot_path}")
        
def run_group_analysis(df, features, output_dir):
    """Runs group analysis based on performance tiers."""
    print("\n--- Approach 2: Group Analysis (High/Mid/Low Performers) ---")
    # Define performance groups based on non_brand_inflow
    df['performance_group'] = pd.qcut(
        df['non_brand_inflow'], 
        q=[0, 0.25, 0.75, 1.0], 
        labels=['Low', 'Mid', 'High'],
        duplicates='drop'
    )
    
    # Calculate mean of features for each group
    group_summary = df.groupby('performance_group')[features + ['non_brand_inflow']].mean().T
    
    print("Group Characteristics (Mean Values):")
    print(group_summary)
    
    # Save results
    path = os.path.join(output_dir, 'performance_group_summary.csv')
    group_summary.to_csv(path, encoding='utf-8-sig')
    print(f"\nGroup summary saved to {path}")

    # Visualize the differences
    df_plot = group_summary.drop('non_brand_inflow').reset_index()
    df_plot = pd.melt(df_plot, id_vars='index', value_vars=['Low', 'Mid', 'High'], var_name='group', value_name='mean_value')

    plt.figure(figsize=(15, 20))
    sns.catplot(data=df_plot, x='mean_value', y='index', hue='group', kind='bar', orient='h', height=8, aspect=1.5)
    plt.title('Comparison of Feature Means Across Performance Groups')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'performance_group_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Group comparison plot saved to {plot_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    paths = setup_environment(base_dir)
    df_master = load_and_prepare_data(paths)
    
    features = [
        'posting_score_numeric', 'exposure_score', 'score_가독성', 'score_주제', 
        'score_콘텐츠', 'score_키워드', 'score_형태소', 'score_후기/정보', 
        'content_with_space', 'word_count', 'valid_images', 'invalid_images'
    ]
    
    # Run regression for both targets
    run_regression_analysis(df_master, features, 'non_brand_inflow', paths['output_dir'])
    run_regression_analysis(df_master, features, 'non_brand_average_C', paths['output_dir'])
    
    # Run group analysis
    run_group_analysis(df_master, features, paths['output_dir'])
    
    print("\n--- Comprehensive Analysis Complete ---")

if __name__ == "__main__":
    main() 