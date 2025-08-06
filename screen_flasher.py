# Screen Flasher Script
#
# Description:
# This script demonstrates how to create a bright, flashing effect that covers
# the entire screen. It uses the built-in Tkinter library. This can be integrated
# into another script to act as a visual alert.
#
# Dependencies:
# - Tkinter (usually included with Python)

import tkinter as tk
import time

class ScreenFlasher:
    def __init__(self):
        """
        Initializes the flasher by creating a top-level, borderless,
        fullscreen window.
        """
        self.root = tk.Tk()
        # Make the window borderless and always on top.
        self.root.overrideredirect(True)
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # Set geometry to cover the whole screen
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        # Make the window stay on top of all other windows
        self.root.wm_attributes("-topmost", 1)
        # Start with the window hidden
        self.root.withdraw()

    def flash(self, duration_sec=2, flashes_per_sec=5):
        """
        Flashes the screen by rapidly showing and hiding the white window.

        Args:
            duration_sec (int): Total time in seconds to flash the screen.
            flashes_per_sec (int): How many times to flash on and off per second.
        """
        print(f"Flashing screen for {duration_sec} seconds...")
        flash_interval = 1.0 / (flashes_per_sec * 2) # Time for one on/off cycle
        end_time = time.time() + duration_sec

        while time.time() < end_time:
            # Turn screen white
            self.root.deiconify() # Show the window
            self.root.configure(bg='white')
            self.root.update()
            time.sleep(flash_interval)

            # Turn screen "off" (hide the window)
            self.root.withdraw() # Hide the window
            self.root.update()
            time.sleep(flash_interval)
        
        print("Flashing finished.")

    def close(self):
        """
        Properly closes the Tkinter window.
        """
        self.root.destroy()

# --- Example Usage ---
if __name__ == "__main__":
    # This part runs only when you execute this script directly.
    print("Starting screen flasher demo in 3 seconds...")
    time.sleep(3)

    flasher = ScreenFlasher()
    try:
        # Flash the screen for 3 seconds
        flasher.flash(duration_sec=3, flashes_per_sec=4)
    finally:
        # Ensure the window is closed even if there's an error
        flasher.close()

