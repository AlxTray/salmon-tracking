import time
from inference_sdk import InferenceHTTPClient
import sys
import cv2
import csv


project_id = "salmon-dnwyp"
model_version = sys.argv[1]

client = InferenceHTTPClient(
     api_url="http://localhost:9001",
     api_key="pTqt5atPorx55y1rXXiC",
)

with open("results/latency.csv", mode="a", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    for i in range(300):
        start = time.time()
        predictions = client.infer(cv2.imread(sys.argv[2]), model_id=f"{project_id}/{model_version}")["predictions"]
        end = time.time()

        latency = end - start
        csv_writer.writerow([sys.argv[1], latency])
