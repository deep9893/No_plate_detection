import cv2
import numpy as np
import os

# Read input video
video_path = 'D:\\Codebook\\No_plate_detection\\demo.mp4' # Replace with the path of your video file
cap = cv2.VideoCapture(video_path)

# Read haarcascade for number plate detection
cascade = cv2.CascadeClassifier("D:\\Study\\haarcascades\\haarcascade_russian_plate_number.xml")

# Create a folder to store captured number plates if it doesn't exist
folder_path = "D:\\Codebook\\No_plate_detection\\CapturedPlates"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Counter for generating unique names
counter = 1

while True:
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no frames are read or end of the video is reached

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license number plates
    plates = cascade.detectMultiScale(gray, 1.2, 5)

    # Loop over all plates
    for (x, y, w, h) in plates:
        # Draw bounding rectangle around the license number plate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        gray_plates = gray[y:y+h, x:x+w]
        color_plates = frame[y:y+h, x:x+w]

        # Save number plate detected with a unique name
        image_name = os.path.join(folder_path, 'Numberplate_' + str(counter) + '.jpg')
        cv2.imwrite(image_name, gray_plates)
        counter += 1

    # Display the frame with bounding boxes
    cv2.imshow('Number Plate Image', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
