import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# Create VideoCapture object
video_path = 'videos/salmon-cut-1.mp4'
cap = cv2.VideoCapture(video_path)

# Create a 3D plot for center of mass
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# CSV file path
csv_file_path = 'results/3d-coordinates.csv'

# Check if the results directory exists, create it if not
if not os.path.exists('results'):
    os.makedirs('results')

project_id = "salmon-dnwyp"
model_version = 11

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="pTqt5atPorx55y1rXXiC",
)

ret, frame = cap.read()
prev_predictions = None

while ret:
    # Open CSV file for writing/appending
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        predictions = client.infer(frame, model_id=f"{project_id}/{model_version}")["predictions"]
        predictions = sorted(predictions, key=lambda x: x["x"])

        if prev_predictions is not None:
            for prediction, prev_prediction in zip(predictions, prev_predictions):

                x = int(prediction["x"])
                y = int(prediction["y"])
                w = int(prediction["width"])
                h = int(prediction["height"])

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cx_prev = int(prev_prediction["x"] + (prev_prediction["width"] / 2))
                cy_prev = int(prev_prediction["y"] + (prev_prediction["height"] / 2))
                cx_current = int(x + (w / 2))
                cy_current = int(y + (h / 2))

                distance = np.linalg.norm(np.array((cx_current - cy_current)) - np.array((cx_prev - cy_prev)))

                cv2.putText(frame, str(distance), (cx_current, cy_current + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

                # Draw a circle at the center of mass
                cv2.circle(frame, (cx_current, cy_current), 5, (255, 0, 0), -1)

                # Plot the center of mass in the 3D plot
                ax.scatter(cx_current, cy_current, cap.get(cv2.CAP_PROP_POS_FRAMES), c='r', marker='o')

                # Write 3D coordinates to CSV file
                csv_writer.writerow([cx_current, cy_current, cap.get(cv2.CAP_PROP_POS_FRAMES)])


    # Display the result
    cv2.imshow('Detected Salmon and their Speed', frame)

    # Update the 3D plot
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Frame Number')
    ax.set_title('Center of Mass in 3D')

    plt.pause(0.001)

    
    prev_predictions = predictions
    ret, frame = cap.read()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object, close the CSV file, and close the window
cap.release()
cv2.destroyAllWindows()
