import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def print_header(title):
    print("\n" + "=" * 60)
    print(f"{title.center(60)}")
    print("=" * 60 + "\n")

def main():
    print_header("Propensity Score Matching (PSM)")
    
    # 1. Create a biased observational dataset
    # Scenario: Did a "Premium AI Feature" (Treatment) increase workflow "Productivity" (Outcome)?
    # Bias: Only rich, highly-experienced developers bought the Premium feature.
    np.random.seed(42)
    n_samples = 1000
    
    # Confounders: Age, Years of Experience
    age = np.random.normal(35, 10, n_samples)
    experience = age / 2 + np.random.normal(0, 2, n_samples)
    
    # Treatment Assignment (Heavily weighted towards older/experienced people)
    # This proves the study is NOT a randomized A/B test!
    treatment_prob = 1 / (1 + np.exp(-(-5 + 0.1 * age + 0.3 * experience)))
    treatment = np.random.binomial(1, treatment_prob)
    
    # Outcome: Productivity Score
    # The treatment actually adds +5 to productivity. Experience also adds to productivity.
    productivity = 50 + (10 * experience) + (5 * treatment) + np.random.normal(0, 5, n_samples)
    
    df_obs = pd.DataFrame({
        'Age': age,
        'Experience': experience,
        'Treatment': treatment, # 1 = Bought Premium, 0 = Did not buy
        'Productivity': productivity
    })
    
    print("WARNING: This is Observational Data. Notice how the Treatment group is heavily biased!")
    print(df_obs.groupby('Treatment')[['Age', 'Experience']].mean())
    
    # 2. Calculate Propensity Scores using Logistic Regression
    print("\n[Step 1] Training Logistic Regression to find Propensity Scores...")
    # Predicting 'Treatment' based on Confounders ('Age', 'Experience')
    features = ['Age', 'Experience']
    ps_model = LogisticRegression()
    ps_model.fit(df_obs[features], df_obs['Treatment'])
    
    # The propensity score is the statistical probability they WOULd HAVE received the treatment
    df_obs['Propensity_Score'] = ps_model.predict_proba(df_obs[features])[:, 1]
    
    print("Propensity Scores successfully mathematically assigned to all 1000 users.")
    
    # 3. Matching
    print("\n[Step 2] Matching Users using K-Nearest Neighbors...")
    treatment_group = df_obs[df_obs['Treatment'] == 1]
    control_group = df_obs[df_obs['Treatment'] == 0]
    
    # Fit KNN on the control group's propensity scores
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(control_group[['Propensity_Score']])
    
    # Find the closest matching control user for every treatment user
    distances, indices = knn.kneighbors(treatment_group[['Propensity_Score']])
    
    # Extract the matched control group
    matched_control_group = control_group.iloc[indices.flatten()]
    
    # Combine the new perfectly balanced dataset!
    matched_data = pd.concat([treatment_group, matched_control_group])
    
    print_header("The Magic of PSM: Bias Eliminated")
    print("Look at how perfectly balanced age and experience are in the matched dataset!")
    print(matched_data.groupby('Treatment')[['Age', 'Experience']].mean())
    
    # 4. Calculate True Causal Effect
    true_effect = matched_data[matched_data['Treatment']==1]['Productivity'].mean() - \
                  matched_data[matched_data['Treatment']==0]['Productivity'].mean()
    
    print(f"\nFinal Causal Inference: The True Impact of the 'Premium Feature' is roughly +{true_effect:.2f} Productivity points.")
    print("We proved this using observational data by mathematically mimicking an A/B test!")

if __name__ == "__main__":
    main()
