import requests
import json
import time

def main():
    print("="*60)
    print(" Simulating an API Client Request ".center(60))
    print("="*60)
    
    # The URL where our Flask API is listening
    URL = "http://localhost:5000/predict"
    
    # Create the payload exactly matching the 4 features the model was trained on
    test_payload = {
        "Feature_A": 0.521,
        "Feature_B": -1.234,
        "Feature_C": 0.056,
        "Feature_D": 2.119
    }
    
    print(f"\n[1] Formulating JSON Payload:")
    print(json.dumps(test_payload, indent=4))
    
    print(f"\n[2] Pinging Server at {URL} via HTTP POST...")
    time.sleep(1) # Dramatic pause
    
    try:
        # Send the request over the local network
        response = requests.post(URL, json=test_payload)
        
        # Check if the server responded positively (HTTP 200 OK)
        if response.status_code == 200:
            print("\n✅ Server Responded Successfully (HTTP 200)!")
            print("\n[3] AI Decision:")
            
            # Parse the JSON response
            result = response.json()
            print(json.dumps(result, indent=4))
            
            print(f"\nConclusion: The AI believes this data point belongs to Class [{result['prediction']}] with {result['confidence']*100}% confidence.")
        else:
            print(f"\n❌ Error! Server responded with HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CRITICAL: Could not connect to the server!")
        print("Did you forget to start the Flask Server in the other terminal?")
        print("Run `python 02_flask_api_server.py` in a separate command prompt first.")

if __name__ == "__main__":
    main()
