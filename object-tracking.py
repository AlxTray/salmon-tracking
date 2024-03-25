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

while ret:
    # Open CSV file for writing/appending
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        predictions = client.infer(frame, model_id=f"{project_id}/{model_version}")["predictions"]

        for prediction in predictions:

            cx = int(prediction["x"])
            cy = int(prediction["y"])
            w = int(prediction["width"])
            h = int(prediction["height"])

            x0 = cx - w / 2
            x1 = cx + w / 2
            y0 = cy - h / 2
            y1 = cy + h / 2
            box_start = (int(x0), int(y0))
            box_end = (int(x1), int(y1))
                
            # Draw bounding box
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)

            # Draw a circle at the center of mass
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Plot the center of mass in the 3D plot
            ax.scatter(cx, cy, cap.get(cv2.CAP_PROP_POS_FRAMES), c='r', marker='o')

            # Write 3D coordinates to CSV file
            csv_writer.writerow([cx, cy, cap.get(cv2.CAP_PROP_POS_FRAMES)])


    # Display the result
    cv2.imshow('Detected Salmon and their Speed', frame)

    # Update the 3D plot
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Frame Number')
    ax.set_title('Center of Mass in 3D')

    plt.pause(0.001)


    ret, frame = cap.read()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object, close the CSV file, and close the window
cap.release()
cv2.destroyAllWindows()
