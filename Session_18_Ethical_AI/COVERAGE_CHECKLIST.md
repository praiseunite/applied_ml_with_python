# Session 18 — Topic Coverage Checklist

## TL18 Topics (Ethical Considerations in AI)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | Define Bias in AI and its types | `README.md` Part 1 (Detailed breakdowns of Historical, Representation, and Measurement/Proxy Bias). | ✅ |
| 2 | Describe Bias uses in AI practices | `README.md` Amazon Hiring example and `notebooks/` Proxy Loan application sandbox. | ✅ |
| 3 | Explain responsible AI practices | `code/01_detecting_bias.py` (Testing the 80% Rule mathematically) and `code/02_mitigating_bias.py` (Responsible removal of proxy structures). | ✅ |

### Modification Note:
As requested by the user, we explicitly avoided embedding buggy third-party Fairness packages (like AIF360/Fairlearn). Instead, students are taught how to calculate the **Disparate Impact ratio** mathematically using raw, stable `pandas` functionality to mathematically guarantee an accessible and bug-free educational experience.
