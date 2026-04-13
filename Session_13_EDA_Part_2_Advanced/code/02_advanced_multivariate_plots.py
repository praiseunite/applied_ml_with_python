import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("\nAdvanced Exploratory Data Analysis (Multivariate)")
    print("Generating Seaborn advanced visualizations...\n")
    
    # We will use the built-in 'penguins' dataset which has multiple correlated numeric features
    df = sns.load_dataset("penguins").dropna()
    
    sns.set_theme(style="whitegrid", context="paper")
    
    # --- 1. Pairplot (The Ultimate Multi-Variate EDA Tool) ---
    print("1. Generating Pairplot (Saved to pairplot.png)...")
    # This single line graphs every numeric variable against every other numeric variable!
    # 'hue' colors the dots by species, instantly showing us clustering patterns!
    pair_fig = sns.pairplot(df, hue="species", corner=True, palette="Dark2")
    pair_fig.figure.suptitle("Advanced Pairplot: Finding Cross-Feature Clusters Instantly", y=1.02, weight='bold')
    pair_fig.savefig("pairplot.png", dpi=200, bbox_inches='tight')
    
    # --- 2. JointPlot (Relationship + Dispersion) ---
    print("2. Generating JointPlot (Saved to jointplot.png)...")
    # Shows the scatter relationship AND the marginal distributions
    joint_fig = sns.jointplot(
        data=df, 
        x="flipper_length_mm", 
        y="body_mass_g", 
        hue="species",
        kind="scatter",
        palette="Dark2",
        alpha=0.7
    )
    joint_fig.figure.suptitle("JointPlot: Relationship + Individual Dispersions", y=1.02, weight='bold')
    joint_fig.savefig("jointplot.png", dpi=200, bbox_inches='tight')

    # --- 3. FacetGrid (Conditioning on multiple variables) ---
    print("3. Generating FacetGrid (Saved to facetgrid.png)...")
    # FacetGrid allows mapping dataset structure onto multiple axes
    g = sns.FacetGrid(df, col="island", hue="species", palette="Dark2", height=4)
    g.map(sns.scatterplot, "flipper_length_mm", "bill_length_mm", alpha=.7)
    g.add_legend()
    g.figure.suptitle("FacetGrid: Breaking relationships down by 'Island'", y=1.05, weight='bold')
    g.savefig("facetgrid.png", dpi=200, bbox_inches='tight')
    
    print("\n✅ All advanced EDA visualizations generated successfully in the current directory.")
    print("Review them to see how we analyze 3+ variables simultaneously without complex math!")

if __name__ == "__main__":
    main()
