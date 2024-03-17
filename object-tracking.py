import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# Create VideoCapture object
video_path = 'videos/salmon-fixed.mp4'
cap = cv2.VideoCapture(video_path)

# Create background subtractor using GSOC
bs_gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()

# Create a 3D plot for center of mass
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# CSV file path
csv_file_path = 'results/3d-coordinates.csv'

# Check if the results directory exists, create it if not
if not os.path.exists('results'):
    os.makedirs('results')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction using GSOC
    fg_mask_gsoc = bs_gsoc.apply(frame)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask_gsoc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Open CSV file for writing/appending
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        for contour in contours:
            # Compute area and center of mass for each contour
            area = cv2.contourArea(contour)
            if 1000 <= area <= 20000:
                # Draw bounding box if the area constraints are met
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Compute the center of mass
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw a circle at the center of mass
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    # Plot the center of mass in the 3D plot
                    ax.scatter(cx, cy, cap.get(cv2.CAP_PROP_POS_FRAMES), c='r', marker='o')

                    # Write 3D coordinates to CSV file
                    csv_writer.writerow([cx, cy, cap.get(cv2.CAP_PROP_POS_FRAMES)])

    # Resize both images to half their original size
    frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    fg_mask_gsoc_resized = cv2.resize(cv2.cvtColor(fg_mask_gsoc, cv2.COLOR_GRAY2BGR), (0, 0), fx=0.5, fy=0.5)

    # Create a side-by-side view of the resized colored image and GSOC result
    result_display = np.hstack((frame_resized, fg_mask_gsoc_resized))

    # Display the result
    cv2.imshow('Resized Colored Image vs GSOC Result', result_display)

    # Update the 3D plot
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Frame Number')
    ax.set_title('Center of Mass in 3D')

    plt.pause(0.001)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object, close the CSV file, and close the window
cap.release()
cv2.destroyAllWindows()
