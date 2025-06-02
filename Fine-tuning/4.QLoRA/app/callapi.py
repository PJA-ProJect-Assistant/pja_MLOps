import requests
import json
import time

# Define the API endpoint
url = "http://localhost:8000/v1/generateText"

headers = {"Content-Type": "application/json"}
data = {"prompt": "Who is president of US?"}

start_time = time.time()
# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")

print("LLM response: " + response.text)