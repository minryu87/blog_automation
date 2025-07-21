import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def analyze_overall_strengths_and_weaknesses():
    """
    Analyzes feature_differences.csv to find the average differences
    for all features, identifying overall strengths and weaknesses.
    """
    # --- 1. Define File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_processed_dir = os.path.join(script_dir, '..', 'data', 'data_processed')
    diff_path = os.path.join(data_processed_dir, 'feature_differences.csv')

    if not os.path.exists(diff_path):
        print(f"Error: feature_differences.csv not found at {diff_path}")
        return

    # --- 2. Load Data ---
    df_diff = pd.read_csv(diff_path)

    # --- 3. Calculate Average Differences ---
    # Select only columns ending with '_diff'
    diff_cols = [col for col in df_diff.columns if col.endswith('_diff')]
    
    # Calculate the mean for each diff column, ignoring NaN values
    average_diffs = df_diff[diff_cols].mean().sort_values()

    # --- 4. Print Results ---
    print("--- Overall Blog Strengths and Weaknesses (Our Average vs. Competitor Average) ---\n")
    print("A negative value means we are lagging behind competitors on average.")
    print("A positive value means we are outperforming competitors on average.\n")
    
    print(average_diffs.to_string())

    # --- 5. Visualize Results ---
    try:
        # Set Korean font
        # For MacOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        if os.path.exists(font_path):
            fm.FontProperties(fname=font_path)
            plt.rc('font', family='AppleGothic')
        else: # For Windows
            font_path = 'c:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                 plt.rc('font', family='Malgun Gothic')
        
        plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign not showing

        plt.figure(figsize=(12, 10))
        average_diffs.plot(kind='barh', color=(average_diffs > 0).map({True: 'skyblue', False: 'salmon'}))
        plt.title('블로그 전체 강점 및 약점 분석 (자사 평균 vs 경쟁사 평균)')
        plt.xlabel('평균 차이 (0보다 크면 강점, 작으면 약점)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save the plot
        plot_path = os.path.join(data_processed_dir, 'overall_diff_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight')
        
        print(f"\n\n[INFO] Analysis visualization saved to:\n{plot_path}")

    except Exception as e:
        print(f"\n[WARN] Could not generate visualization. Error: {e}")
        print("[INFO] Please ensure you have a Korean font (AppleGothic or Malgun Gothic) installed.")


if __name__ == '__main__':
    analyze_overall_strengths_and_weaknesses() 