from inference_sdk import InferenceHTTPClient
import sys
import cv2

project_id = "salmon-dnwyp"
model_version = sys.argv[1]

client = InferenceHTTPClient(
     api_url="http://localhost:9001",
     api_key="pTqt5atPorx55y1rXXiC",
)

image = cv2.imread(sys.argv[2])

predictions = client.infer(image, model_id=f"{project_id}/{model_version}")["predictions"]

for prediction in predictions:
    cx = int(prediction["x"])
    cy = int(prediction["y"])
    w = int(prediction["width"])
    h = int(prediction["height"])

    x1 = cx - w / 2
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2

    box_start = (int(x1), int(y1))
    box_end = (int(x2), int(y2))

    cv2.rectangle(image, box_start, box_end, (0, 255, 0), 1)

cv2.imshow('Detections for version ' + str(model_version), image)
cv2.waitKey(0)
cv2.destroyAllWindows()
