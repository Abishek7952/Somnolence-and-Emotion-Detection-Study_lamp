# Drowsiness Detection Script
#
# Description:
# This script uses a webcam to monitor a person's eyes and detect signs of drowsiness.
# If the EAR for BOTH eyes falls below a threshold for a sustained period, it triggers
# a sound alarm. The screen flashing logic has been moved to a separate script.
#
# Dependencies:
# - OpenCV: For video capture and image processing.
# - dlib: For face and facial landmark detection.
# - SciPy: For calculating the Euclidean distance between facial landmarks.
# - Pygame: For playing the alert sound.
# - NumPy: For numerical operations.
#
# Setup:
# 1. Make sure you have installed all the required libraries (see setup guide).
# 2. Download the 'shape_predictor_68_face_landmarks.dat' file and place it
#    in the same directory as this script.
# 3. Place an alarm sound file (e.g., 'alarm.wav') in the same directory.

import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time
import numpy as np

# --- Constants and Configuration ---

# Thresholds for drowsiness detection
# If EAR drops below this value, we consider it a blink.
EYE_AR_THRESH = 0.25
# Number of consecutive frames the eye must be below the threshold to trigger the alarm.
EYE_AR_CONSEC_FRAMES = 35

# --- Function to Calculate Eye Aspect Ratio (EAR) ---

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) given the coordinates of the eye landmarks.
    The EAR is the ratio of the distances between the vertical eye landmarks to the
    distance between the horizontal eye landmarks.
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

# --- Initialization ---

print("[INFO] Initializing...")

# Initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# The path to the landmark predictor file.
predictor_path = 'shape_predictor_68_face_landmarks.dat'
try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    print(f"[ERROR] Failed to load landmark predictor model.")
    print(f"[ERROR] Please ensure '{predictor_path}' is in the same directory as the script.")
    print(f"[ERROR] dlib error: {e}")
    exit()


# Grab the indexes of the facial landmarks for the left and right eye
# dlib's 68-point model has specific index ranges for each feature.
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Initialize Pygame for sound alerts
try:
    pygame.mixer.init()
    alarm_sound_path = 'alarm.wav'
    pygame.mixer.music.load(alarm_sound_path)
    print(f"[INFO] Alarm sound '{alarm_sound_path}' loaded successfully.")
except pygame.error as e:
    print(f"[ERROR] Could not load or play the alarm sound: {alarm_sound_path}")
    print(f"[ERROR] Please ensure the file exists and is a valid sound file (e.g., .wav, .mp3).")
    print(f"[ERROR] Pygame error: {e}")
    # We can continue without sound, but we should notify the user.
    alarm_sound_path = None


# Start the video stream from the webcam
print("[INFO] Starting video stream...")
# Use 0 for the default webcam. If you have multiple cameras, you might need to use 1, 2, etc.
cap = cv2.VideoCapture(0)
time.sleep(1.0) # Allow camera to warm up

# Frame counter to track how long the eyes have been closed
frame_counter = 0
alarm_on = False

# --- Main Video Processing Loop ---

while True:
    # Grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame from camera. Exiting.")
        break

    # For better performance, we can resize the frame.
    # A smaller frame processes faster.
    # frame = cv2.resize(frame, (450, 350))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        
        coords = []
        for i in range(0, 68):
            coords.append((shape.part(i).x, shape.part(i).y))

        # Extract the left and right eye coordinates
        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        
        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratio together for both eyes for visualization
        ear = (leftEAR + rightEAR) / 2.0

        # For visualization, draw the contours of the eyes on the frame
        leftEyeHull = cv2.convexHull(np.array(leftEye))
        rightEyeHull = cv2.convexHull(np.array(rightEye))
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the EAR for *both* eyes is below the threshold.
        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            frame_counter += 1

            # If the eyes were closed for a sufficient number of frames,
            # then sound the alarm
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                # If the alarm is not already on, turn it on
                if not alarm_on:
                    alarm_on = True
                    # Play the alarm sound if it was loaded
                    if alarm_sound_path:
                        print("[ALERT] Drowsiness Detected!")
                        pygame.mixer.music.play(-1) # -1 loops the sound

                # Draw an alert on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Otherwise, at least one eye is open, so reset the counter and alarm.
        else:
            frame_counter = 0
            if alarm_on:
                alarm_on = False
                if alarm_sound_path:
                    pygame.mixer.music.stop()

        # Display the computed EARs on the frame for debugging/visualization
        cv2.putText(frame, f"L_EAR: {leftEAR:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"R_EAR: {rightEAR:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
cap.release()
pygame.mixer.quit()