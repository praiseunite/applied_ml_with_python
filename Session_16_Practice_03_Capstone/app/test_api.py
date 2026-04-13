import requests
import json
import os

def main():
    print("="*60)
    print(" Simulating Marketing Dept API Request ".center(60))
    print("="*60)
    
    URL = "http://localhost:7860/analyze_ab_test"
    
    # 1. Load the generated JSON data
    data_path = '../data/raw_ab_data.json'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find {data_path}. Did you run generate_ab_test_data.py?")
        return
        
    with open(data_path, 'r') as f:
        payload = json.load(f)
        
    print(f"[1] Loaded {len(payload)} raw records. Proceeding to ping Statistical Microservice...")
    
    # 2. Push to local API
    try:
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print("\n✅ Web Server Responded Successfully (HTTP 200)!")
            print("\n[The Math Engine Result]:")
            
            result = response.json()
            print(json.dumps(result, indent=4))
            
            print(f"\nFinal Executive Recommendation: {result['Recommendation']}")
            print("Notice that the server executed EDA data cleaning BEFORE running the T-Test!")
            
        else:
            print(f"\n❌ Error! Server responded with HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CRITICAL: Could not connect to the server!")
        print("Run `python app.py` in a separate command prompt first to wake up the Endpoint.")

if __name__ == "__main__":
    main()
