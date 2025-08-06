# Screen Flasher Script
#
# Description:
# This script provides a class to create a fullscreen window that can be
# shown or hidden on command. It's designed to be controlled from a
# main application loop, avoiding threading issues.
#
# Dependencies:
# - Tkinter (usually included with Python)

import tkinter as tk

class ScreenFlasher:
    def __init__(self):
        """
        Initializes the flasher by creating a top-level, borderless,
        fullscreen window.
        """
        self.root = tk.Tk()
        # Make the window borderless and always on top.
        self.root.overrideredirect(True)
        
        # Get screen dimensions and set geometry
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Keep the window on top of all others
        self.root.wm_attributes("-topmost", 1)
        
        # Set background to white once
        self.root.configure(bg='white')
        
        # Start with the window hidden
        self.root.withdraw()
        self.is_showing = False

    def set_flash_state(self, show: bool):
        """
        Shows or hides the window based on the 'show' boolean argument.
        This method is intended to be called repeatedly from a main loop.
        """
        # Show the window if it's supposed to be visible but currently isn't
        if show and not self.is_showing:
            self.root.deiconify()
            self.is_showing = True
        # Hide the window if it's not supposed to be visible but currently is
        elif not show and self.is_showing:
            self.root.withdraw()
            self.is_showing = False
        
        # Process Tkinter events to ensure the window updates correctly
        # when called from within another loop (like OpenCV's).
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        """
        Properly closes the Tkinter window.
        """
        self.root.destroy()

