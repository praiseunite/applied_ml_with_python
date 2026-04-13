import numpy as np
from scipy import stats

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def main():
    print_header("A/B Testing: Frequentist Statistical Analysis")
    
    # 1. Hypothesis: "The New layout (Group B) will increase conversions compared to the Old layout (Group A)."
    print("Hypothesis: Layout B drives significantly more clicks than Layout A.")
    
    # 2. Collect Data (Simulation)
    # We simulate 1000 users. Control group has a 10% conversion rate. Treatment has a 14% conversion rate.
    # 1 = Clicked (Converted), 0 = Did not click
    np.random.seed(42)
    group_A_conversions = np.random.binomial(n=1, p=0.10, size=1000)
    group_B_conversions = np.random.binomial(n=1, p=0.14, size=1000)
    
    # 3. Measurement
    cr_A = group_A_conversions.mean()
    cr_B = group_B_conversions.mean()
    
    print("\n[Measurement Results]")
    print(f"Group A (Control) Conversion Rate:   {cr_A * 100:.1f}%")
    print(f"Group B (Treatment) Conversion Rate: {cr_B * 100:.1f}%")
    print(f"Absolute Lift:                       {(cr_B - cr_A) * 100:.1f}%")
    
    # 4. Significance Analysis (T-Test)
    # Is 14% vs 10% just random luck? We must calculate the p-value.
    print("\n[Running T-Test for Statistical Significance...]")
    
    # We use ttest_ind (independent two-sample t-test)
    t_stat, p_value = stats.ttest_ind(group_B_conversions, group_A_conversions)
    
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.5f}")
    
    # 5. The Business Decision
    print_header("The Business Decision")
    alpha = 0.05 # Standard industry threshold (95% confidence)
    
    if p_value < alpha:
        print(f"✅ STATISTICALLY SIGNIFICANT (p < {alpha}).")
        print("The difference is likely NOT due to random chance.")
        print("Business Action: Deploy Layout B to all users immediately!")
    else:
        print(f"❌ NOT SIGNIFICANT (p >= {alpha}).")
        print("The difference could easily be random luck.")
        print("Business Action: Keep Layout A. Do not waste money deploying the update.")

if __name__ == "__main__":
    main()
