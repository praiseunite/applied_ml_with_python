# Session 14 — Topic Coverage Checklist

## TL14 Topics (Model Deployment and Maintenance)

| # | Required Topic | Covered In | Status |
|---|----------------|------------|--------|
| 1 | Explain model serialization, deserialization, and its applications. | `README.md` Part 1, `notebooks/`, `code/01_train_and_serialize.py` | ✅ |
| 2 | Describe Web API’s and its practices. | `README.md` Part 2, `code/03_api_client_test.py` (HTTP POST) | ✅ |
| 3 | Outline Flask for API development. | `README.md` Part 3, `code/02_flask_api_server.py` | ✅ |
| 4 | Explain Deployment on Cloud Platform. | `README.md` Part 4, `portfolio/portfolio_component.md` (Docker) | ✅ |

### Technical Note:
As requested, we introduced `joblib` over `pickle` as it is the industry standard for serializing `scikit-learn` algorithms. For Cloud Deployment, we mapped the instructions directly to **Hugging Face Docker Spaces** and provided the exact `Dockerfile` configuration, aligning with the Portfolio 5 goal set in the main course outline.
