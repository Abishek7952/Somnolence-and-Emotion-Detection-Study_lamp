# Drowsiness & Emotion Detector
#
# Description:
# This script combines drowsiness detection and emotion analysis.
# - Drowsiness: Triggers a sound alarm and a flashing white screen.
# - Emotion: Detects the user's emotion and displays it, along with a
#   corresponding color swatch to simulate an ambient lamp.
#
# Dependencies:
# - All previous libraries plus 'deepface'.
# - All helper scripts ('screen_flasher.py', 'emotion_detector.py',
#   'color_mapper.py') must be in the same directory.

import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time
import numpy as np

# Import our custom helper modules
from screen_flasher import ScreenFlasher
from emotion_detector import EmotionDetector
from color_mapper import get_color_for_emotion

# --- Constants and Configuration ---
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 35
# How often to run emotion detection (e.g., once every 15 frames)
EMOTION_CHECK_INTERVAL = 15 

# --- Function to Calculate Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Initialization ---
print("[INFO] Initializing...")

# Initialize dlib for drowsiness detection
print("[INFO] Loading facial landmark predictor...")
drowsiness_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
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

# Initialize our custom modules
print("[INFO] Initializing screen flasher...")
flasher = ScreenFlasher()
print("[INFO] Initializing emotion detector...")
emotion_detector = EmotionDetector()

# Start video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Counters and status variables
drowsiness_frame_counter = 0
emotion_check_counter = 0
alarm_on = False
last_detected_emotion = "neutral" # Start with a default emotion

# --- Main Video Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = drowsiness_detector(gray, 0)

    # --- Emotion Detection Logic ---
    # Only check for emotion every N frames to save resources
    if emotion_check_counter % EMOTION_CHECK_INTERVAL == 0:
        # We run analysis on a copy of the frame
        detected_emotion = emotion_detector.analyze_frame(frame.copy())
        if detected_emotion:
            last_detected_emotion = detected_emotion
    emotion_check_counter += 1

    # --- Drowsiness Detection Logic ---
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
            drowsiness_frame_counter += 1
            if drowsiness_frame_counter >= EYE_AR_CONSEC_FRAMES and not alarm_on:
                alarm_on = True
                print("[ALERT] Drowsiness Detected!")
                if alarm_sound_path:
                    pygame.mixer.music.play(-1)
        else:
            drowsiness_frame_counter = 0
            if alarm_on:
                alarm_on = False
                if alarm_sound_path:
                    pygame.mixer.music.stop()
    
    # --- Visual Updates ---
    # Handle screen flashing for drowsiness
    if alarm_on:
        # Simple flash logic: on for 5 frames, off for 5
        if (emotion_check_counter // 5) % 2 == 0:
            flasher.set_flash_state(True)
        else:
            flasher.set_flash_state(False)
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        flasher.set_flash_state(False)

    # Display emotion and color swatch
    emotion_color = get_color_for_emotion(last_detected_emotion)
    # Note: OpenCV uses BGR format, so we reverse the RGB tuple
    bgr_emotion_color = emotion_color[::-1] 
    
    cv2.putText(frame, f"Emotion: {last_detected_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Draw a rectangle to represent the ambient lamp color
    cv2.rectangle(frame, (frame.shape[1] - 60, 10), (frame.shape[1] - 10, 60), bgr_emotion_color, -1)

    # Show the final frame
    cv2.imshow("Emotion-Detecting Study Lamp", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
cap.release()
pygame.mixer.quit()
flasher.close()
