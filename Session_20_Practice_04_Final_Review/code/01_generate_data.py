# Session 20 — Script 01: Generate Synthetic HR Attrition Dataset
# ================================================================
# This script generates a realistic employee attrition dataset
# for the Session 20 Final Capstone Project.
#
# The dataset is intentionally designed with:
# - Class imbalance (only ~20% of employees leave)
# - Missing values (simulating incomplete HR records)
# - Proxy bias risk (salary correlates with gender due to historical pay gaps)
# - Outliers (a few employees with extreme overtime hours)
#
# Dependencies: pip install pandas numpy

import numpy as np
import pandas as pd
import os

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def main():
    print_header("Generating HR Employee Attrition Dataset")
    
    np.random.seed(42)
    n = 1500
    
    # ─── Demographics ────────────────────────────────────────────────────
    employee_id = [f"EMP-{i:04d}" for i in range(1, n + 1)]
    age = np.random.normal(36, 9, n).clip(22, 62).astype(int)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.58, 0.42])
    
    # Department distribution
    department = np.random.choice(
        ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'],
        n, p=[0.30, 0.22, 0.15, 0.10, 0.13, 0.10]
    )
    
    # ─── Employment Features ─────────────────────────────────────────────
    years_at_company = np.random.exponential(4, n).clip(0.5, 30).round(1)
    
    # Distance from home (km)
    distance_from_home = np.random.exponential(10, n).clip(1, 80).astype(int)
    
    # Monthly salary — with a realistic (and problematic) gender pay gap
    base_salary = np.random.normal(5500, 1500, n).clip(2500, 15000)
    # Inject historical bias: males get ~8% more on average
    gender_multiplier = np.where(gender == 'Male', 1.08, 1.0)
    monthly_salary = (base_salary * gender_multiplier).round(0).astype(int)
    
    # Overtime hours per month
    overtime_hours = np.random.exponential(5, n).clip(0, 30).astype(int)
    # Inject ~2% extreme outliers (employees doing 50-80 hours overtime)
    outlier_mask = np.random.random(n) < 0.02
    overtime_hours[outlier_mask] = np.random.randint(50, 80, outlier_mask.sum())
    
    # Job satisfaction (1-5 scale)
    job_satisfaction = np.random.choice([1, 2, 3, 4, 5], n, p=[0.08, 0.15, 0.30, 0.30, 0.17])
    
    # Performance rating (1-5 scale, slightly correlated with satisfaction)
    performance_base = job_satisfaction + np.random.normal(0, 0.8, n)
    performance_rating = np.clip(np.round(performance_base), 1, 5).astype(int)
    
    # Number of promotions in last 5 years
    promotions_last_5yr = np.random.choice([0, 1, 2, 3], n, p=[0.45, 0.35, 0.15, 0.05])
    
    # Work-life balance (1-4, where 4 is best)
    work_life_balance = np.random.choice([1, 2, 3, 4], n, p=[0.10, 0.25, 0.40, 0.25])
    
    # Number of companies worked before
    num_companies_before = np.random.poisson(2, n).clip(0, 9)
    
    # ─── Generate Attrition Target (Imbalanced ~20%) ─────────────────────
    # Attrition risk increases with:
    #   - Low satisfaction, low salary, high overtime, long commute
    #   - Few promotions, low work-life balance, many previous companies
    attrition_score = (
        -0.4 * job_satisfaction          # Lower satisfaction → higher risk
        - 0.0002 * monthly_salary        # Lower salary → higher risk
        + 0.02 * overtime_hours          # More overtime → higher risk
        + 0.01 * distance_from_home      # Longer commute → higher risk
        - 0.3 * promotions_last_5yr      # Fewer promotions → higher risk
        - 0.2 * work_life_balance        # Worse balance → higher risk
        + 0.15 * num_companies_before    # Job hoppers → higher risk
        - 0.05 * years_at_company        # Shorter tenure → higher risk (less invested)
        + np.random.normal(0, 0.8, n)    # Random noise
    )
    
    # Normalize to probability and set ~20% attrition rate
    attrition_prob = 1 / (1 + np.exp(-(attrition_score - np.percentile(attrition_score, 80))))
    attrition = np.random.binomial(1, attrition_prob)
    
    # ─── Inject Missing Values (~5% across select columns) ───────────────
    df = pd.DataFrame({
        'EmployeeID': employee_id,
        'Age': age.astype(float),
        'Gender': gender,
        'Department': department,
        'YearsAtCompany': years_at_company,
        'DistanceFromHome': distance_from_home.astype(float),
        'MonthlySalary': monthly_salary.astype(float),
        'OvertimeHours': overtime_hours.astype(float),
        'JobSatisfaction': job_satisfaction.astype(float),
        'PerformanceRating': performance_rating.astype(float),
        'PromotionsLast5Yr': promotions_last_5yr.astype(float),
        'WorkLifeBalance': work_life_balance.astype(float),
        'NumCompaniesBefore': num_companies_before.astype(float),
        'Attrition': attrition
    })
    
    # Inject NaN values in specific columns (simulating incomplete HR records)
    missing_cols = ['Age', 'MonthlySalary', 'OvertimeHours', 'JobSatisfaction', 'PerformanceRating']
    for col in missing_cols:
        missing_mask = np.random.random(n) < 0.05
        df.loc[missing_mask, col] = np.nan
    
    # ─── Save Dataset ────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'employee_attrition.csv')
    df.to_csv(output_path, index=False)
    
    # ─── Dataset Summary ─────────────────────────────────────────────────
    print(f"  Dataset saved to: {output_path}")
    print(f"  Total employees: {len(df)}")
    print(f"  Features: {len(df.columns) - 2} (excluding EmployeeID and Attrition)")
    print(f"\n  Class Distribution:")
    print(f"    Stayed (0): {(df['Attrition'] == 0).sum()} ({(df['Attrition'] == 0).mean():.1%})")
    print(f"    Left   (1): {(df['Attrition'] == 1).sum()} ({(df['Attrition'] == 1).mean():.1%})")
    
    print(f"\n  Missing Values:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"    {col}: {missing[col]} missing ({missing[col]/len(df):.1%})")
    
    print(f"\n  Gender Distribution:")
    print(f"    Male:   {(df['Gender'] == 'Male').sum()}")
    print(f"    Female: {(df['Gender'] == 'Female').sum()}")
    
    print(f"\n  Salary by Gender (mean):")
    print(f"    Male:   ${df[df['Gender'] == 'Male']['MonthlySalary'].mean():,.0f}")
    print(f"    Female: ${df[df['Gender'] == 'Female']['MonthlySalary'].mean():,.0f}")
    print(f"    (Note: This pay gap is intentionally injected for the ethical audit exercise)")
    
    print(f"\n  Overtime Outliers (>45 hrs):")
    outliers = df[df['OvertimeHours'] > 45]
    print(f"    {len(outliers)} employees with extreme overtime")
    
    print(f"\n  Dataset is ready for the pipeline!\n")

if __name__ == "__main__":
    main()
