import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Generating Synthetic Store Dataset...")
    np.random.seed(42)
    
    # Generating 500 rows of realistic business data
    data = {
        'Marketing_Spend': np.random.uniform(1000, 10000, 500),
        'Store_Size_SqFt': np.random.normal(5000, 1500, 500),
        'Employee_Count': np.random.randint(5, 50, 500),
    }
    
    # Introduce real-world correlations
    # Revenue is heavily influenced by Marketing and somewhat by Store Size
    data['Revenue'] = (data['Marketing_Spend'] * 5.5) + (data['Store_Size_SqFt'] * 1.2) + np.random.normal(0, 5000, 500)
    
    # Introduce a negative correlation
    # Distance to competitor negatively impacts revenue
    data['Distance_To_Competitor_km'] = np.random.uniform(0.5, 20.0, 500)
    data['Revenue'] -= data['Distance_To_Competitor_km'] * 1000
    
    df = pd.DataFrame(data)
    
    print("\nVisualizing Patterns for Optimal Presentation...")
    
    # Create an optimal visual presentation area with Seaborn
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Correlation Heatmap
    # Essential for finding underlying patterns instantly
    ax1 = plt.subplot(2, 2, 1)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax1)
    ax1.set_title("1. Feature Correlation Heatmap", weight='bold')
    
    # 2. Scatter Plot with Trendline (Positive Pattern)
    ax2 = plt.subplot(2, 2, 2)
    sns.regplot(data=df, x='Marketing_Spend', y='Revenue', scatter_kws={'alpha':0.5, 'color':'#3498db'}, line_kws={'color':'#e74c3c'}, ax=ax2)
    ax2.set_title("2. Marketing Spend vs Revenue", weight='bold')
    ax2.set_xlabel("Marketing Spend ($)")
    
    # 3. Scatter Plot with Trendline (Negative Pattern)
    ax3 = plt.subplot(2, 2, 3)
    sns.regplot(data=df, x='Distance_To_Competitor_km', y='Revenue', scatter_kws={'alpha':0.5, 'color':'#9b59b6'}, line_kws={'color':'#e74c3c'}, ax=ax3)
    ax3.set_title("3. Effect of Threat (Competitor Proximity)", weight='bold')
    ax3.set_xlabel("Distance to Competitor (km)")
    
    # 4. Boxplot (Detecting Data Distributions & Irregularities visually)
    ax4 = plt.subplot(2, 2, 4)
    # Add a synthetic outlier to prove boxplots catch them
    df.loc[0, 'Revenue'] = df['Revenue'].max() * 2 
    sns.boxplot(data=df, y='Revenue', color='#2ecc71', ax=ax4)
    ax4.set_title("4. Revenue Distribution (Note the Outlier!)", weight='bold')
    
    # Optimal Presentation Layout
    plt.tight_layout(pad=3.0)
    
    # Save the figure to disk instead of requiring an interactive display
    # This ensures the code always works in CI pipelines and terminal execution.
    output_path = "eda_patterns_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualizations successfully generated and saved to: {output_path}")
    print("Open the image to review the underlying trends discovered in the data!")

if __name__ == "__main__":
    main()
