# Drowsiness Detection Script (No Threading)
#
# Description:
# This script uses a webcam to detect drowsiness. When detected, it plays a
# sound alarm and triggers a screen-flashing effect by controlling the
# ScreenFlasher class within the main loop, avoiding threading errors.
#
# Dependencies:
# - opencv-python, dlib, scipy, pygame, numpy
# - screen_flasher.py (must be in the same directory)

import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time
import numpy as np
from screen_flasher import ScreenFlasher # Import the flasher

# --- Constants and Configuration ---
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 35

# --- Function to Calculate Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Initialization ---
print("[INFO] Initializing...")

# Initialize dlib
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Get eye landmark indexes
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Initialize Pygame for sound
try:
    pygame.mixer.init()
    alarm_sound_path = 'alarm.wav'
    pygame.mixer.music.load(alarm_sound_path)
    print(f"[INFO] Alarm sound '{alarm_sound_path}' loaded.")
except pygame.error as e:
    print(f"[ERROR] Could not load alarm sound: {e}")
    alarm_sound_path = None

# Initialize the Screen Flasher
print("[INFO] Initializing screen flasher...")
flasher = ScreenFlasher()

# Start video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Counters and alarm status
frame_counter = 0
alarm_on = False
flash_cycle_counter = 0 # Counter to control the flash rate

# --- Main Video Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        coords = [ (shape.part(i).x, shape.part(i).y) for i in range(68) ]

        leftEye = coords[lStart:lEnd]
        rightEye = coords[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        leftEyeHull = cv2.convexHull(np.array(leftEye))
        rightEyeHull = cv2.convexHull(np.array(rightEye))
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH:
            frame_counter += 1
            if frame_counter >= EYE_AR_CONSEC_FRAMES and not alarm_on:
                alarm_on = True
                print("[ALERT] Drowsiness Detected!")
                if alarm_sound_path:
                    pygame.mixer.music.play(-1)
        else:
            frame_counter = 0
            if alarm_on:
                alarm_on = False
                if alarm_sound_path:
                    pygame.mixer.music.stop()
    
    # --- NEW FLASHING LOGIC (IN MAIN THREAD) ---
    if alarm_on:
        # Create a flashing effect by toggling visibility every 10 frames
        # (e.g., show for 5 frames, hide for 5 frames)
        if flash_cycle_counter % 10 < 5:
            flasher.set_flash_state(True)
        else:
            flasher.set_flash_state(False)
        flash_cycle_counter += 1
        # Draw the alert text on the frame
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        flasher.set_flash_state(False) # Ensure window is hidden
        flash_cycle_counter = 0 # Reset the cycle

    # Display EAR values
    cv2.putText(frame, f"L_EAR: {leftEAR:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"R_EAR: {rightEAR:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the video frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
cap.release()
pygame.mixer.quit()
flasher.close()
