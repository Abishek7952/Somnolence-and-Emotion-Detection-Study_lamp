# Color Mapper Script
#
# Description:
# This script provides a simple mapping from a detected emotion string
# to a specific color represented as an RGB tuple.

def get_color_for_emotion(emotion):
    """
    Maps an emotion string to an RGB color tuple.

    Args:
        emotion (str): The detected emotion (e.g., 'happy', 'sad', 'angry').

    Returns:
        A tuple representing an RGB color (R, G, B), or a default color
        if the emotion is not recognized or is None.
    """
    # Define the color mappings
    # These colors are chosen to be calming or mood-lifting.
    color_map = {
        'angry':    (255, 0, 0),      # Red - for clear feedback, though not calming
        'disgust':  (0, 128, 0),      # Green
        'fear':     (128, 0, 128),    # Purple
        'happy':    (255, 255, 0),    # Yellow
        'sad':      (0, 0, 255),      # Blue
        'surprise': (255, 165, 0),    # Orange
        'neutral':  (255, 255, 255)   # White
    }
    
    # Return the color for the given emotion.
    # .get() is used to provide a default value if the emotion is not in the map.
    # The default color here is a soft white.
    return color_map.get(emotion, (200, 200, 200))

# --- Example Usage ---
if __name__ == '__main__':
    # This part runs only when you execute this script directly.
    test_emotions = ['happy', 'sad', 'angry', 'neutral', 'unknown_emotion', None]
    
    print("Testing color mappings:")
    for emotion in test_emotions:
        color = get_color_for_emotion(emotion)
        print(f"Emotion: {emotion}, Mapped Color (RGB): {color}")

