import cv2
import numpy as np
import torch
from inference_sdk import InferenceHTTPClient
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import math

import sys

from deep_sort.deep_sort import DeepSort

video_path = None
cap = None

def videoArgument(index):
    global cap

    video_path = sys.argv[index + 1]
    cap = cv2.VideoCapture(video_path)

def webcamArgument(index):
    global cap

    cap = cv2.VideoCapture(0)

switch_case = {
    '--video': videoArgument,
    '--webcam': webcamArgument
}

for index, argument in enumerate(sys.argv):
    if argument in switch_case:
        switch_case[argument](index)

if cap == None:
    print("Please provide either a '--video *path/to/video*' or '--webcam' switch")
    exit()


# Create a 3D plot for center of mass
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# CSV file path
csv_file_path = 'results/trajectories.csv'

# Check if the results directory exists, create it if not
if not os.path.exists('results'):
    os.makedirs('results')

project_id = "salmon-dnwyp"
model_version = 5

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="pTqt5atPorx55y1rXXiC",
)


tracker = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7")

ret, frame = cap.read()

previous_frame_tracker = None
start_time = None

while ret:
    tracker_output = None

    # Open CSV file for writing/appending
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        predictions = client.infer(frame, model_id=f"{project_id}/{model_version}")["predictions"]
        
        bbox_list = []
        confidence_list = []
        for prediction in predictions:

            cx = int(prediction["x"])
            cy = int(prediction["y"])
            w = int(prediction["width"])
            h = int(prediction["height"])
            bbox_list.append([cx, cy, w, h])

            confidence = prediction["confidence"]
            confidence_list.append(confidence)

            # Draw a circle at the center of mass
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Plot the center of mass in the 3D plot
            ax.scatter(cx, cy, cap.get(cv2.CAP_PROP_POS_FRAMES), c='r', marker='o')
        
        if bbox_list:
            tracker_output = tracker.update(np.array(bbox_list), np.array(confidence_list), frame)
            
            if previous_frame_tracker is not None:
                matching_salmons

                for track in tracker_output:
                    x1, y1, x2, y2, id = track

                    for prev_track in previous_frame_tracker:
                        prev_x1, prev_y1, prev_x2, prev_y2, prev_id = prev_track

                        if id == prev_id:
                            box_start = (int(x1), int(y1))
                            box_end = (int(x2), int(y2))

                            information_box_start = (int(x1), int(y1 - 20))
                            information_box_end = (int(x2), int(y1))
                
                            # Draw bounding box
                            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 1)
                            cv2.rectangle(frame, information_box_end, information_box_start, (0, 255, 0), -1)
                            
                            # Calculate centre point of current frame and previous
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            prev_cx = (prev_x1 + prev_x2) / 2
                            prev_cy = (prev_y1 + prev_y2) / 2
                            
                            # Calculate time passed between frame
                            time_per_frame = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                            #######################################################
                            distance = (cx + cy) - (prev_cx + prev_cy)
                            linear_speed = round((distance / time_per_frame), 2)
                            #######################################################
                            change_in_angle = math.degrees(math.atan2((x2 - prev_x1), (y2 - prev_y1)))
                            angular_speed = round((change_in_angle / time_per_frame), 2)

                            cv2.putText(frame, str(linear_speed) + " pixels/sec", (int(x1 + 175), int(y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
                            cv2.putText(frame, str(angular_speed) + " deg/sec", (int(x1 + 175), int(y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
                            cv2.putText(frame, "Salmon ID: " + str(id), (int(x1 + 5), int(y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)

                            # Write trajectories to CSV file
                            csv_writer.writerow([id, x1, y1, x2, y2, linear_speed, angular_speed])

                            break


    # Display the result
    cv2.imshow('Detected Salmon and their Speed', frame)

    # Update the 3D plot
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Frame Number')
    ax.set_title('Center of Mass in 3D')

    plt.pause(0.001)

    previous_frame_tracker = tracker_output
    start_time = cv2.getTickCount()
    ret, frame = cap.read()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object, close the CSV file, and close the window
cap.release()
cv2.destroyAllWindows()
