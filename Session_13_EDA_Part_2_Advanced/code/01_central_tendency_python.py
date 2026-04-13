import pandas as pd
import numpy as np

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def main():
    print_header("Applied Python Statistical Indicators")
    print("We don't need manual math. Pandas does it all for us!")
    
    # Generate an explicitly skewed dataset (e.g., Salary data)
    # Most people make ~50k, but a few make 500k+
    np.random.seed(42)
    normal_salaries = np.random.normal(50000, 10000, 95)
    ceo_salaries = np.array([500000, 750000, 1000000, 2000000, 5000000])
    
    all_salaries = np.concatenate([normal_salaries, ceo_salaries])
    
    # Store in a DataFrame
    df = pd.DataFrame({'Salary': all_salaries})
    
    # 1. Central Tendency
    print_header("1. Central Tendency")
    mean_val = df['Salary'].mean()
    median_val = df['Salary'].median()
    
    print(f"Mean (Average):   ${mean_val:,.2f}")
    print(f"Median (Middle):  ${median_val:,.2f}")
    print(f"\nInsight: Notice how extreme the difference is! If you tell an ML model the 'Average' person here makes ${mean_val:,.0f}, it will fail terribly. For finding the 'True Center' of skewed data, use the Median.")
    
    # 2. Dispersion Metrics
    print_header("2. Dispersion Metrics")
    std_dev = df['Salary'].std()
    variance = df['Salary'].var()
    data_range = df['Salary'].max() - df['Salary'].min()
    
    print(f"Standard Deviation: ${std_dev:,.2f}")
    print(f"Data Range:         ${data_range:,.2f}")
    print(f"\nInsight: The standard deviation is massive. The data is incredibly spread out.")

    # 3. Shape / Distribution Indicators (Skewness and Kurtosis)
    print_header("3. Shape Indicators (Advanced Analysis)")
    skewness = df['Salary'].skew()
    kurtosis = df['Salary'].kurtosis()
    
    print(f"Skewness: {skewness:.2f}")
    if skewness > 1:
        print("  -> Data is highly Right-Skewed (tail extends to the right). Use Log-Transformation before ML!")
        
    print(f"\nKurtosis: {kurtosis:.2f}")
    if kurtosis > 3:
        print("  -> Data has heavy tails (Leptokurtic). Lots of extreme outliers are present!")
        
    # 4. The Magic summary command
    print_header("4. The Standard Describe Command")
    print(df.describe())

if __name__ == "__main__":
    main()
