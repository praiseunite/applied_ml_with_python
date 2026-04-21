# Session 20 — Script 02: Full End-to-End ML Pipeline
# ====================================================
# This is the INTEGRATIVE CAPSTONE PROJECT that combines techniques from
# at least 8 different sessions into a single, cohesive pipeline.
#
# Pipeline Steps:
#   1. Load & Explore Data            (Sessions 12-13: EDA)
#   2. Handle Missing Values           (Session 12: Imputation)
#   3. Detect & Handle Outliers        (Session 13: IQR method)
#   4. Encode Categorical Features     (Sessions 1-2: Feature Engineering)
#   5. Handle Class Imbalance          (Session 7: SMOTE)
#   6. Train Ensemble Models           (Sessions 9-10: Random Forest, XGBoost)
#   7. Evaluate with Proper Metrics    (Session 10: ROC-AUC, Classification Report)
#   8. Explain Model Predictions       (Sessions 17, 19: Feature Importance)
#   9. Ethical Audit                   (Session 18: Disparate Impact by Gender)
#  10. Document Search                 (Session 19: TF-IDF query over HR policies)
#
# Dependencies: pip install scikit-learn pandas numpy imbalanced-learn

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def print_subheader(title):
    print(f"\n  --- {title} ---\n")

def main():
    print_header("SESSION 20: Full End-to-End ML Pipeline")
    print("  Employee Attrition Prediction — Integrating 8+ Sessions\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD & EXPLORE DATA (Sessions 12-13: EDA)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 1: Exploratory Data Analysis (Sessions 12-13)")
    
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'employee_attrition.csv')
    
    if not os.path.exists(data_path):
        print("  Dataset not found! Generating it now...")
        import subprocess
        gen_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '01_generate_data.py')
        subprocess.run(['python', gen_script], check=True)
    
    df = pd.read_csv(data_path)
    
    print(f"  Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\n  Column Types:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"    {col:25s} | {str(dtype):10s} | {non_null}/{len(df)} non-null")
    
    print_subheader("Class Distribution (Attrition)")
    attrition_counts = df['Attrition'].value_counts()
    for label, count in attrition_counts.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        label_str = "Stayed" if label == 0 else "Left"
        print(f"    {label_str} ({label}): {count:5d} ({pct:.1f}%) {bar}")
    
    imbalance_ratio = attrition_counts[0] / attrition_counts[1]
    print(f"\n    Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"    >> This is a CLASS IMBALANCE problem! (Session 7)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: HANDLE MISSING VALUES (Session 12: Imputation)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 2: Handling Missing Values (Session 12)")
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print("  Missing values detected:\n")
        for col, count in missing_cols.items():
            print(f"    {col:25s}: {count:3d} missing ({count/len(df):.1%})")
        
        # Strategy: Impute with MEDIAN (robust to outliers)
        print("\n  Strategy: Median imputation (robust to the outliers we know exist)")
        for col in missing_cols.index:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    Filled '{col}' with median = {median_val:.1f}")
        
        total_remaining = df.isnull().sum().sum()
        print(f"\n  Remaining missing values: {total_remaining}")
    else:
        print("  No missing values found.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: DETECT & HANDLE OUTLIERS (Session 13: IQR Method)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 3: Outlier Detection & Treatment (Session 13)")
    
    numerical_cols = ['OvertimeHours', 'MonthlySalary', 'DistanceFromHome']
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        n_outliers = len(outliers)
        
        if n_outliers > 0:
            print(f"  {col}: {n_outliers} outliers detected (IQR range: [{lower:.1f}, {upper:.1f}])")
            # Strategy: CAP outliers at the boundary (winsorization)
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"    >> Capped at [{lower:.1f}, {upper:.1f}]")
        else:
            print(f"  {col}: No outliers detected (IQR range: [{lower:.1f}, {upper:.1f}])")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: ENCODE CATEGORICAL FEATURES (Sessions 1-2: Feature Engineering)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 4: Feature Engineering (Sessions 1-2)")
    
    # Save gender mapping for ethical audit later
    gender_mapping = {'Male': 1, 'Female': 0}
    df['Gender_Code'] = df['Gender'].map(gender_mapping)
    print(f"  Encoded 'Gender': {gender_mapping}")
    
    # One-hot encode Department
    dept_dummies = pd.get_dummies(df['Department'], prefix='Dept', drop_first=True)
    df = pd.concat([df, dept_dummies], axis=1)
    print(f"  One-hot encoded 'Department': {list(dept_dummies.columns)}")
    
    # Define features (exclude ID, raw Gender, raw Department)
    exclude = ['EmployeeID', 'Gender', 'Department', 'Attrition']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].values
    y = df['Attrition'].values
    
    print(f"\n  Final feature set: {len(feature_cols)} features")
    for i, col in enumerate(feature_cols):
        print(f"    [{i:2d}] {col}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: HANDLE CLASS IMBALANCE (Session 7: SMOTE)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 5: Handling Class Imbalance (Session 7)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")
    print(f"  Train class balance: {np.mean(y_train):.1%} attrition")
    
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\n  SMOTE applied!")
        print(f"    Before SMOTE: {len(X_train)} samples ({np.mean(y_train):.1%} attrition)")
        print(f"    After SMOTE:  {len(X_train_balanced)} samples ({np.mean(y_train_balanced):.1%} attrition)")
    except ImportError:
        print("\n  imbalanced-learn not installed. Using class_weight='balanced' instead.")
        print("  Install with: pip install imbalanced-learn")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: TRAIN ENSEMBLE MODELS (Sessions 9-10)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 6: Training Ensemble Models (Sessions 9-10)")
    
    # Model A: Random Forest (Bagging)
    print_subheader("Model A: Random Forest (Bagging)")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    
    print(f"  Accuracy: {rf_acc * 100:.1f}%")
    print(f"  ROC-AUC:  {rf_auc:.3f}")
    
    # Model B: Gradient Boosting (Boosting)
    print_subheader("Model B: Gradient Boosting (Boosting)")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    )
    gb_model.fit(X_train_balanced, y_train_balanced)
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_proba)
    
    print(f"  Accuracy: {gb_acc * 100:.1f}%")
    print(f"  ROC-AUC:  {gb_auc:.3f}")
    
    # Select best model
    best_model_name = "Random Forest" if rf_auc >= gb_auc else "Gradient Boosting"
    best_model = rf_model if rf_auc >= gb_auc else gb_model
    best_pred = rf_pred if rf_auc >= gb_auc else gb_pred
    best_proba = rf_proba if rf_auc >= gb_auc else gb_proba
    best_auc = max(rf_auc, gb_auc)
    
    print(f"\n  >> Selected: {best_model_name} (ROC-AUC = {best_auc:.3f})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: EVALUATION (Session 10: Confusion Matrix, Classification Report)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 7: Model Evaluation (Session 10)")
    
    print("  Classification Report:")
    print(classification_report(y_test, best_pred, target_names=['Stayed', 'Left']))
    
    cm = confusion_matrix(y_test, best_pred)
    print("  Confusion Matrix:")
    print(f"                  Predicted STAY  Predicted LEAVE")
    print(f"    Actual STAY       {cm[0][0]:5d}           {cm[0][1]:5d}")
    print(f"    Actual LEAVE      {cm[1][0]:5d}           {cm[1][1]:5d}")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5, scoring='roc_auc')
    print(f"\n  5-Fold Cross-Validation ROC-AUC:")
    for i, score in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {score:.3f}")
    print(f"    Mean:  {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: EXPLAINABILITY (Sessions 17, 19: Feature Importance)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 8: Model Explainability (Sessions 17 & 19)")
    
    importances = best_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("  Top 10 Features Driving Attrition:\n")
    print(f"    {'Rank':<6} {'Feature':<25} {'Importance':>12}")
    print(f"    {'-'*45}")
    
    for rank, idx in enumerate(sorted_idx[:10], 1):
        bar = "#" * int(importances[idx] * 100)
        print(f"    {rank:<6} {feature_cols[idx]:<25} {importances[idx]:>10.4f}  {bar}")
    
    print("\n  HR Interpretation:")
    top_feature = feature_cols[sorted_idx[0]]
    print(f"    The strongest predictor of attrition is '{top_feature}'.")
    print(f"    HR should focus retention programs on employees with extreme values in this feature.")
    
    # Individual prediction explanation
    print_subheader("Explaining a Specific Employee's Prediction")
    
    # Find an employee predicted to leave
    leave_indices = np.where(best_pred == 1)[0]
    if len(leave_indices) > 0:
        sample_idx = leave_indices[0]
        sample_data = X_test[sample_idx]
        sample_pred = best_model.predict([sample_data])[0]
        sample_proba = best_model.predict_proba([sample_data])[0]
        
        print(f"  Employee at test index #{sample_idx}:")
        print(f"  Prediction: {'WILL LEAVE' if sample_pred == 1 else 'WILL STAY'}")
        print(f"  Confidence: Stay = {sample_proba[0]:.1%}, Leave = {sample_proba[1]:.1%}")
        print(f"\n  Key feature values for this employee:")
        
        for idx in sorted_idx[:5]:
            print(f"    {feature_cols[idx]:25s} = {sample_data[idx]:.1f}")
        
        print(f"\n  >> These are the factors HR should address for this individual.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: ETHICAL AUDIT (Session 18: Disparate Impact by Gender)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 9: Ethical Audit (Session 18)")
    
    print("  Question: Does the model predict 'Will Leave' at a fair rate")
    print("  across gender groups? Or is it biased?\n")
    
    # Get Gender_Code from the test set features
    gender_col_idx = feature_cols.index('Gender_Code')
    test_genders = X_test[:, gender_col_idx]
    
    # Male (1) predictions
    male_mask = test_genders == 1
    male_leave_rate = best_pred[male_mask].mean()
    
    # Female (0) predictions
    female_mask = test_genders == 0
    female_leave_rate = best_pred[female_mask].mean()
    
    print(f"  AI 'Will Leave' Prediction Rate:")
    print(f"    Male employees:   {male_leave_rate * 100:.1f}%")
    print(f"    Female employees: {female_leave_rate * 100:.1f}%")
    
    # Disparate Impact: Compare prediction rates
    # Here we check if one group is disproportionately flagged as "at risk"
    if male_leave_rate > 0 and female_leave_rate > 0:
        if male_leave_rate >= female_leave_rate:
            di_ratio = female_leave_rate / male_leave_rate
            privileged = "Male"
            unprivileged = "Female"
        else:
            di_ratio = male_leave_rate / female_leave_rate
            privileged = "Female"
            unprivileged = "Male"
        
        print(f"\n  Disparate Impact Ratio ({unprivileged} / {privileged}): {di_ratio:.3f}")
        
        if di_ratio < 0.80:
            print(f"  FAILED: Ratio below 0.80 (Four-Fifths Rule).")
            print(f"  The model disproportionately flags {unprivileged} employees as at-risk.")
            print(f"  Investigation needed: Is 'MonthlySalary' acting as a PROXY for Gender?")
            print(f"  (Remember: We injected a gender pay gap in the data generator!)")
        else:
            print(f"  PASSED: Ratio is >= 0.80. The model treats both groups fairly.")
    
    print(f"\n  Proxy Bias Check:")
    print(f"    'MonthlySalary' correlates with 'Gender_Code' by design.")
    print(f"    If the model heavily uses MonthlySalary (check Step 8 rankings),")
    print(f"    it may be indirectly discriminating by gender through the pay gap proxy.")
    
    salary_importance = importances[feature_cols.index('MonthlySalary')]
    gender_importance = importances[feature_cols.index('Gender_Code')]
    print(f"\n    MonthlySalary importance: {salary_importance:.4f}")
    print(f"    Gender_Code importance:   {gender_importance:.4f}")
    
    if salary_importance > 0.10:
        print(f"    >> WARNING: MonthlySalary is a top predictor and carries gender proxy risk.")
    else:
        print(f"    >> MonthlySalary has moderate/low importance. Proxy risk is contained.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: DOCUMENT QUERY SYSTEM (Session 19: TF-IDF)
    # ═══════════════════════════════════════════════════════════════════════
    print_header("Step 10: HR Policy Search Engine (Session 19)")
    
    print("  Bonus: A TF-IDF search engine over HR policy documents.")
    print("  HR managers can ask questions to find relevant company policies.\n")
    
    hr_policies = [
        "Employees showing low job satisfaction scores should be referred to the Employee Engagement program. "
        "Quarterly pulse surveys must be conducted. Managers receive automated alerts when team satisfaction drops below 3.0.",
        
        "Retention bonuses of up to 15% of annual salary may be offered to high-performing employees in critical roles "
        "identified as flight risks. Approval from VP and HR Business Partner required. Bonuses vest over 12 months.",
        
        "Overtime must not exceed 20 hours per month without director approval. Employees averaging above 15 hours "
        "overtime for 3 consecutive months must be flagged for workload review. Mandatory rest periods apply.",
        
        "Remote work and flexible hours are available to improve work-life balance. Employees with commutes exceeding "
        "45 minutes are eligible for 2 additional remote days per week upon manager approval.",
        
        "Exit interviews must be conducted within 5 business days of resignation notice. Results are anonymized and "
        "aggregated quarterly to identify systemic issues. All exit interview data feeds the attrition prediction model.",
        
        "Career development plans must be reviewed bi-annually. Employees with fewer than 1 promotion in 5 years "
        "should receive accelerated mentorship. Internal mobility postings are prioritized for tenured employees.",
    ]
    
    policy_titles = [
        "Employee Engagement & Satisfaction Monitoring",
        "Retention Bonus Program",
        "Overtime Management Policy",
        "Remote Work & Commute Flexibility",
        "Exit Interview Protocol",
        "Career Development & Promotion Pathways",
    ]
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(hr_policies)
    
    demo_queries = [
        "How do we keep our best employees from quitting?",
        "What should we do about employees working too many extra hours?",
        "How can we help employees who live far from the office?",
    ]
    
    for query in demo_queries:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        best_idx = sims.argmax()
        
        print(f"  Query: \"{query}\"")
        print(f"    Best Match: {policy_titles[best_idx]} ({sims[best_idx]*100:.1f}% confidence)")
        print(f"    Preview:    \"{hr_policies[best_idx][:100]}...\"\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print_header("PIPELINE COMPLETE: Final Summary")
    
    print(f"  {'Step':<5} {'Phase':<35} {'Session(s)':<15} {'Status':<10}")
    print(f"  {'-'*65}")
    print(f"  {'1':<5} {'Exploratory Data Analysis':<35} {'12-13':<15} {'DONE':<10}")
    print(f"  {'2':<5} {'Missing Value Imputation':<35} {'12':<15} {'DONE':<10}")
    print(f"  {'3':<5} {'Outlier Detection (IQR)':<35} {'13':<15} {'DONE':<10}")
    print(f"  {'4':<5} {'Feature Engineering':<35} {'1-2':<15} {'DONE':<10}")
    print(f"  {'5':<5} {'Class Imbalance (SMOTE)':<35} {'7':<15} {'DONE':<10}")
    print(f"  {'6':<5} {'Ensemble Models (RF + GB)':<35} {'9-10':<15} {'DONE':<10}")
    print(f"  {'7':<5} {'Evaluation (AUC + CV)':<35} {'10':<15} {'DONE':<10}")
    print(f"  {'8':<5} {'Explainability':<35} {'17, 19':<15} {'DONE':<10}")
    print(f"  {'9':<5} {'Ethical Audit (Disparate Impact)':<35} {'18':<15} {'DONE':<10}")
    print(f"  {'10':<5} {'Document Query (TF-IDF)':<35} {'19':<15} {'DONE':<10}")
    
    print(f"\n  Best Model: {best_model_name}")
    print(f"  ROC-AUC:    {best_auc:.3f}")
    print(f"  CV Mean:    {cv_scores.mean():.3f}")
    
    print(f"\n  Congratulations! You have completed the full Applied ML curriculum.")
    print(f"  This pipeline demonstrates mastery of the complete ML lifecycle.\n")

if __name__ == "__main__":
    main()
