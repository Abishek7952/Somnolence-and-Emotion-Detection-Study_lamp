# Emotion Detector Script
#
# Description:
# This script uses the 'deepface' library to analyze a video frame (as a NumPy array)
# and determine the dominant emotion of the face present in the frame.
#
# Dependencies:
# - deepface (which includes tensorflow, opencv-python, and numpy)

from deepface import DeepFace
import logging

# Optional: Suppress the detailed, often lengthy, logging from deepface
# You can comment this out if you want to see the backend messages.
logging.getLogger('deepface').setLevel(logging.ERROR)


class EmotionDetector:
    """
    A class to detect emotions from an image frame.
    """
    def __init__(self):
        """
        The constructor for the EmotionDetector class.
        DeepFace handles its own model loading on the first run, so
        we don't need to initialize much here.
        """
        pass

    def analyze_frame(self, frame):
        """
        Analyzes a single video frame to detect the dominant emotion.

        Args:
            frame: A NumPy array representing the video frame (in BGR format from OpenCV).

        Returns:
            A string representing the dominant emotion (e.g., 'happy', 'sad', 'neutral'),
            or None if no face is detected or an error occurs.
        """
        try:
            # DeepFace's analyze function can detect emotion, age, gender, and race.
            # We specify that we only want to analyze 'emotion'.
            # It returns a list of dictionaries, one for each face found.
            analysis_result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=True, # Ensures it doesn't try to analyze a frame with no face
                silent=True # Hides the progress bar for a cleaner output
            )
            
            # The result is a list, we'll take the first face found.
            if analysis_result and len(analysis_result) > 0:
                # The dominant emotion is stored in the 'dominant_emotion' key.
                dominant_emotion = analysis_result[0]['dominant_emotion']
                return dominant_emotion
            else:
                return None

        except Exception as e:
            # DeepFace will raise an exception if it cannot find a face in the image.
            # We catch this and return None, so our main program doesn't crash.
            # print(f"No face detected or error in analysis: {e}")
            return None

# --- Example Usage ---
if __name__ == '__main__':
    # This part runs only when you execute this script directly.
    # It's useful for testing the emotion detector with your webcam.
    import cv2
    
    detector = EmotionDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting emotion detection test. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze the frame to get the emotion
        emotion = detector.analyze_frame(frame)
        
        # Display the detected emotion on the frame
        if emotion:
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Emotion: N/A", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
        cv2.imshow("Emotion Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
