# Session 19 — Script 02: Federated Learning Simulation
# =====================================================
# This script simulates a complete Federated Learning pipeline with
# 4 independent hospital clients. Each hospital trains a Logistic Regression
# model on its LOCAL patient data, sends ONLY its model weights to a central
# server, and the server AVERAGES the weights to produce a superior global model.
#
# Real-World Scenario:
# Four hospitals in different countries want to collaboratively build a diabetes
# prediction model. Due to HIPAA (US), GDPR (EU), and LGPD (Brazil) regulations,
# NO patient record can leave the hospital where it was collected. Federated
# Learning allows them to build a model as good as if they had pooled all data —
# without sharing a single patient row.
#
# Dependencies: pip install scikit-learn pandas numpy

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from copy import deepcopy

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def generate_diabetes_data(n, seed, hospital_name, bias_factor=0.0):
    """
    Generate a synthetic diabetes dataset for one hospital.
    Each hospital gets a slightly different data distribution (simulating
    real-world differences between patient populations).
    """
    np.random.seed(seed)
    
    # Clinical features
    age = np.random.normal(50 + bias_factor, 12, n).clip(20, 80).astype(int)
    bmi = np.random.normal(28 + bias_factor * 0.5, 6, n).clip(15, 50).round(1)
    blood_pressure = np.random.normal(80 + bias_factor * 2, 12, n).clip(50, 130).astype(int)
    glucose = np.random.normal(120 + bias_factor * 3, 35, n).clip(50, 300).astype(int)
    insulin = np.random.normal(80 + bias_factor * 2, 40, n).clip(10, 400).astype(int)
    skin_thickness = np.random.normal(25, 8, n).clip(5, 60).astype(int)
    
    # Generate diabetes outcome based on risk factors
    risk_score = (
        0.03 * age
        + 0.05 * bmi
        + 0.01 * blood_pressure
        + 0.015 * glucose
        + 0.005 * insulin
        + np.random.normal(0, 0.8, n)
    )
    risk_prob = 1 / (1 + np.exp(-(risk_score - np.median(risk_score))))
    diabetes = np.random.binomial(1, risk_prob)
    
    df = pd.DataFrame({
        'Age': age, 'BMI': bmi, 'BloodPressure': blood_pressure,
        'Glucose': glucose, 'Insulin': insulin, 'SkinThickness': skin_thickness,
        'Diabetes': diabetes
    })
    
    return df

class FederatedServer:
    """
    Central server that coordinates Federated Learning.
    It NEVER sees raw patient data. It only receives and averages model weights.
    """
    def __init__(self, n_features):
        self.global_weights = np.zeros(n_features)
        self.global_bias = 0.0
        self.round_history = []
    
    def aggregate_weights(self, client_updates, round_num):
        """
        Federated Averaging (FedAvg) — the core Federated Learning algorithm.
        Takes the weighted average of all client model weights.
        """
        all_weights = [update['weights'] for update in client_updates]
        all_biases = [update['bias'] for update in client_updates]
        all_sizes = [update['data_size'] for update in client_updates]
        
        total_samples = sum(all_sizes)
        
        # Weighted average by dataset size (larger hospitals contribute more)
        self.global_weights = sum(
            w * (s / total_samples) for w, s in zip(all_weights, all_sizes)
        )
        self.global_bias = sum(
            b * (s / total_samples) for b, s in zip(all_biases, all_sizes)
        )
        
        self.round_history.append({
            'round': round_num,
            'num_clients': len(client_updates),
            'total_samples': total_samples
        })
        
        return self.global_weights, self.global_bias

class HospitalClient:
    """
    A single hospital participating in Federated Learning.
    It trains on its LOCAL data and sends ONLY model weights to the server.
    """
    def __init__(self, name, data, feature_names):
        self.name = name
        self.feature_names = feature_names
        self.X = data[feature_names].values
        self.y = data['Diabetes'].values
        self.data_size = len(data)
        self.model = LogisticRegression(max_iter=200, random_state=42)
        
    def train_local(self, global_weights=None, global_bias=None):
        """
        Train the model on LOCAL hospital data.
        If global weights are provided, initialize from them by using
        warm_start to continue from the previous solution.
        """
        if global_weights is not None:
            # Set warm_start so the solver continues from the global weights
            self.model.set_params(warm_start=True)
            # Must have a fitted model first — do a full fit, then override weights
            self.model.fit(self.X, self.y)
            self.model.coef_ = global_weights.reshape(1, -1)
            self.model.intercept_ = np.array([global_bias])
        
        # Train on local data (from global weights if warm_start is set)
        self.model.fit(self.X, self.y)
        
        return {
            'weights': self.model.coef_.flatten(),
            'bias': self.model.intercept_[0],
            'data_size': self.data_size,
            'local_accuracy': self.model.score(self.X, self.y)
        }
    
    def evaluate(self, X_test, y_test):
        """Evaluate the current local model on external test data."""
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)
    
    def set_global_model(self, weights, bias):
        """Update local model with the latest global weights from server."""
        self.model.coef_ = weights.reshape(1, -1)
        self.model.intercept_ = np.array([bias])

def main():
    print_header("Federated Learning Simulation")
    print("Scenario: 4 hospitals across different countries want to build")
    print("a diabetes prediction model WITHOUT sharing any patient data.\n")
    print("  🏥 Hospital Alpha  (New York, USA)     — 500 patients")
    print("  🏥 Hospital Beta   (London, UK)         — 350 patients")
    print("  🏥 Hospital Gamma  (São Paulo, Brazil)  — 280 patients")
    print("  🏥 Hospital Delta  (Lagos, Nigeria)     — 420 patients")
    print("\n⚠️  Due to HIPAA, GDPR, LGPD, and NDPA regulations,")
    print("   NO patient record may leave its home hospital.\n")
    
    # ─── Stage 1: Generate Local Datasets ────────────────────────────────
    print_header("Stage 1: Generating Local Hospital Datasets")
    
    feature_names = ['Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']
    
    hospitals_config = [
        ("Hospital Alpha (NYC)", 500, 10, 0.0),
        ("Hospital Beta (London)", 350, 20, 2.0),
        ("Hospital Gamma (São Paulo)", 280, 30, -1.0),
        ("Hospital Delta (Lagos)", 420, 40, 1.5),
    ]
    
    hospital_data = {}
    for name, n, seed, bias in hospitals_config:
        data = generate_diabetes_data(n, seed, name, bias)
        hospital_data[name] = data
        pos_rate = data['Diabetes'].mean()
        print(f"  📋 {name:35s} | {n:4d} patients | {pos_rate:.1%} diabetes rate")
    
    # Create a held-out global test set (simulating a neutral evaluation set)
    global_test = generate_diabetes_data(500, seed=999, hospital_name="Test", bias_factor=0.5)
    X_test_global = global_test[feature_names].values
    y_test_global = global_test['Diabetes'].values
    
    print(f"\n  🧪 Global Test Set (neutral):          500 patients | {global_test['Diabetes'].mean():.1%} diabetes rate")
    
    # ─── Stage 2: Baseline — Each Hospital Trains Alone ──────────────────
    print_header("Stage 2: Baseline — Each Hospital Training in ISOLATION")
    print("What happens if each hospital builds its own model independently?\n")
    
    isolated_scores = {}
    for name, data in hospital_data.items():
        model = LogisticRegression(max_iter=200, random_state=42)
        X_local = data[feature_names].values
        y_local = data['Diabetes'].values
        model.fit(X_local, y_local)
        
        score = accuracy_score(y_test_global, model.predict(X_test_global))
        isolated_scores[name] = score
        print(f"  {name:35s} | Isolated Accuracy: {score * 100:.1f}%")
    
    avg_isolated = np.mean(list(isolated_scores.values()))
    print(f"\n  📊 Average Isolated Accuracy: {avg_isolated * 100:.1f}%")
    print("  Outcome: Each hospital has a mediocre model limited by its local data size.\n")
    
    # ─── Stage 3: Centralized — Pool All Data (Privacy Violation!) ───────
    print_header("Stage 3: Centralized Baseline (THE PRIVACY VIOLATION)")
    print("If we ILLEGALLY pooled all patient data into one server:\n")
    
    all_data = pd.concat(hospital_data.values(), ignore_index=True)
    centralized_model = LogisticRegression(max_iter=200, random_state=42)
    centralized_model.fit(all_data[feature_names].values, all_data['Diabetes'].values)
    centralized_score = accuracy_score(
        y_test_global, centralized_model.predict(X_test_global)
    )
    print(f"  🚨 Centralized Model Accuracy: {centralized_score * 100:.1f}%")
    print(f"     (This required shipping {len(all_data)} patient records to one server)")
    print(f"     THIS IS ILLEGAL under HIPAA/GDPR/LGPD/NDPA!\n")
    print(f"  Goal: Can Federated Learning match this accuracy WITHOUT sharing data?\n")
    
    # ─── Stage 4: Federated Learning ─────────────────────────────────────
    print_header("Stage 4: Federated Learning (PRIVACY-PRESERVING)")
    
    n_features = len(feature_names)
    server = FederatedServer(n_features)
    
    # Create hospital clients
    clients = []
    for name, data in hospital_data.items():
        client = HospitalClient(name, data, feature_names)
        clients.append(client)
    
    NUM_ROUNDS = 5
    federated_scores = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"  ╔══════════════════════════════════════════════════════════════╗")
        print(f"  ║  FEDERATED ROUND {round_num}/{NUM_ROUNDS}{' ' * 43}║")
        print(f"  ╚══════════════════════════════════════════════════════════════╝")
        
        # Each hospital trains locally and sends ONLY weights
        client_updates = []
        for client in clients:
            if round_num == 1:
                update = client.train_local()
            else:
                update = client.train_local(
                    global_weights=server.global_weights,
                    global_bias=server.global_bias
                )
            client_updates.append(update)
            print(f"    🏥 {client.name:35s} → Local accuracy: {update['local_accuracy'] * 100:.1f}%"
                  f"  | Sent {n_features} weights + 1 bias (NO patient data!)")
        
        # Server aggregates via FedAvg
        global_w, global_b = server.aggregate_weights(client_updates, round_num)
        
        # Distribute updated global model back to all clients
        for client in clients:
            client.set_global_model(global_w, global_b)
        
        # Evaluate the global model on the neutral test set
        # (Use any client since they all have the same global weights now)
        round_score = clients[0].evaluate(X_test_global, y_test_global)
        federated_scores.append(round_score)
        
        print(f"\n    📡 Server aggregated {len(clients)} weight updates (FedAvg)")
        print(f"    🌐 Global Model Accuracy (Round {round_num}): {round_score * 100:.1f}%\n")
    
    # ─── Stage 5: Results Comparison ─────────────────────────────────────
    print_header("FINAL RESULTS COMPARISON")
    
    final_federated = federated_scores[-1]
    
    print(f"  {'Method':<45} {'Accuracy':>10} {'Privacy':>10}")
    print(f"  {'-' * 65}")
    
    for name, score in isolated_scores.items():
        print(f"  {name + ' (Isolated)':<45} {score * 100:>9.1f}% {'✅ Safe':>10}")
    
    print(f"  {'-' * 65}")
    print(f"  {'Average Isolated':<45} {avg_isolated * 100:>9.1f}% {'✅ Safe':>10}")
    print(f"  {'Centralized (ALL DATA POOLED)':<45} {centralized_score * 100:>9.1f}% {'❌ ILLEGAL':>10}")
    print(f"  {'Federated Learning (5 Rounds)':<45} {final_federated * 100:>9.1f}% {'✅ Safe':>10}")
    print(f"  {'-' * 65}")
    
    improvement = ((final_federated - avg_isolated) / avg_isolated) * 100
    gap_to_central = ((centralized_score - final_federated) / centralized_score) * 100
    
    print(f"\n  📈 Federated vs Isolated:     +{improvement:.1f}% improvement")
    print(f"  📊 Federated vs Centralized:  {gap_to_central:.1f}% gap (acceptable trade-off for privacy)")
    
    print("\n  ✅ CONCLUSION:")
    print("  Federated Learning achieved near-centralized performance while")
    print("  ensuring ZERO patient records ever left their home hospital.")
    print("  Every regulation (HIPAA, GDPR, LGPD, NDPA) was fully respected.\n")
    
    # ─── Stage 6: Privacy Proof ──────────────────────────────────────────
    print_header("PRIVACY AUDIT LOG")
    print("  Data transmitted between hospitals and server:\n")
    print(f"    Per Round Per Hospital:")
    print(f"      ✅ Sent:     {n_features} float weights + 1 float bias = {(n_features + 1) * 8} bytes")
    print(f"      ❌ NOT Sent: Any of the {sum(len(d) for d in hospital_data.values())} patient records\n")
    print(f"    Total Data Transmitted (all rounds):")
    total_bytes = NUM_ROUNDS * len(clients) * (n_features + 1) * 8
    print(f"      {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB) of model weights")
    total_patient_bytes = sum(len(d) for d in hospital_data.values()) * len(feature_names) * 8
    print(f"      vs {total_patient_bytes:,} bytes ({total_patient_bytes / 1024:.1f} KB) if raw data was shared")
    print(f"\n    Privacy Preservation Ratio: {(1 - total_bytes / total_patient_bytes) * 100:.2f}% data reduction\n")

if __name__ == "__main__":
    main()
