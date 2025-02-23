"""
Blinky Eye Blink Detector
=========================
A lightweight Python application that detects eye blinks using OpenCV and dlib,
with adjustable sensitivity and alert delay settings.

Author: Nick Rotundo  
License: MIT  
GitHub: https://github.com/nickjrotundo/blinky  

---------------------
Requirements:
---------------------
- Python 3.8+  
- Required pip packages (install via `pip install -r requirements.txt`):  
    pip install opencv-python dlib numpy winsound imutils  

- Download Required Model:  
    You must download `shape_predictor_68_face_landmarks.dat` from:  
    https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat  
    Place it in the same directory as `blinky.py`.  

---------------------
How to Build as an Executable:
---------------------
To package this as a standalone `.exe` (Windows):  
    pyinstaller --onefile --noconsole --add-data "shape_predictor_68_face_landmarks.dat;." blinky.py  

This will generate `dist/blinky.exe`, which can be run without Python installed.  

---------------------
Usage:
---------------------
1. Run `blinky.py` or `blinky.exe`  
2. Adjust sensitivity with the "Trigger" slider (higher = more sensitive)  
3. Adjust alert delay with the "Delay(s)" slider (higher = longer wait before alert)  
4. Press `H` for help, `Q` to quit  

---------------------
Notes:
---------------------
- If the camera selection takes a few seconds, it is auto-detecting the first valid camera. Virtual cameras can cause this to take longer.
- The application stores slider settings in `blinkyconfig.json` for convenience on subsequent runs.
- If packaging with PyInstaller, ensure the `.dat` file is included (see command above).  
"""

import cv2
import dlib
import numpy as np
import time
import threading
import winsound
import json
import sys
import os
from imutils import face_utils

# Determine correct path for the .dat file
if getattr(sys, 'frozen', False):  # Running from PyInstaller .exe
    BASE_PATH = sys._MEIPASS
else:  # Running as a script
    BASE_PATH = os.path.abspath(".")

# Need to have the .dat file in the same directory as the script (or update the referece)
# https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat to download.
DAT_FILE = os.path.join(BASE_PATH, "shape_predictor_68_face_landmarks.dat")

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DAT_FILE)  # Ensure the correct path is used

# Indices for eye landmarks
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Config file for storing last used settings
CONFIG_FILE = "blinkyconfig.json"

def load_config():
    """Loads stored slider values from a config file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Invalid config file. Using defaults.")
    return {"blink_threshold": 26, "alert_delay": 9}  # Default values

def save_config():
    """Saves current slider values to a config file."""
    config_data = {"blink_threshold": blink_threshold, "alert_delay": alert_delay_slider}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f)

# Load stored settings
config = load_config()
blink_threshold = config["blink_threshold"]
alert_delay_slider = config["alert_delay"]
show_help = False  # Help screen toggle
blink_time = time.time()
alert_active = False  # Prevent multiple alerts from overlapping

# Function to auto-detect the first camera that sees a face. It's....ok, for the most part. A little clunky. Fine if camera is the first one for sure, or if no virtual cameras that can cause a hang.
# Tested working with a Logitech C920 variant, as well as with the same camera but through ManyCam as a virtual webcam.
def auto_detect_camera():
    cv2.namedWindow("Camera Search")  # Temporary window for feedback

    for i in range(10):  # Try the first 10 cameras
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue  # Skip unavailable cameras

        start_time = time.time()

        for _ in range(10):  # Read a few frames to ensure detection
            ret, frame = cap.read()
            if not ret:
                if time.time() - start_time > 2:  # Timeout after 2 seconds
                    break
                continue

            # Show progress in temporary window
            frame_display = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(frame_display, "Finding Camera...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_display, f"Checking camera {i}...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Camera Search", frame_display)
            cv2.waitKey(100)  # Small delay so the window updates

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                cap.release()
                cv2.destroyWindow("Camera Search")  # Close temp window
                print(f"Using camera index {i}")
                return i  # Return the working camera index

        cap.release()

    cv2.destroyWindow("Camera Search")  # Close temp window
    print("No valid camera with a face detected! Defaulting to camera 0.")
    return 0  # Default to the first camera if no face is detected

# Start webcam with auto-detected camera
camera_index = auto_detect_camera()

# Show immediate feedback before opening the main window
cv2.namedWindow("Starting...")
starting_frame = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.putText(starting_frame, "Starting...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow("Starting...", starting_frame)
cv2.waitKey(100)

cap = cv2.VideoCapture(camera_index)
cv2.destroyWindow("Starting...")

# Callback functions to update sliders and save config
def update_threshold(val):
    global blink_threshold
    blink_threshold = val
    save_config()

def update_delay(val):
    global alert_delay_slider
    alert_delay_slider = val
    save_config()

# Create OpenCV window with trackbars
cv2.namedWindow("Blinky Eye Blink Detector")
cv2.createTrackbar("Trigger", "Blinky Eye Blink Detector", blink_threshold, 30, update_threshold) 
cv2.createTrackbar("Delay(s)", "Blinky Eye Blink Detector", alert_delay_slider, 30, update_delay)

# Set min/max for trackbar. This overrides the 30 above fyi. 
cv2.setTrackbarMin("Trigger", "Blinky Eye Blink Detector", 20)
cv2.setTrackbarMax("Trigger", "Blinky Eye Blink Detector", 30)

def play_alert():
    global alert_active
    winsound.Beep(1000, 500)  # Play a simple beep. I was using playsound for an mp3, but it would sometimes not work because...Windows reasons? pygame, etc., could be another option.
    alert_active = False  # Reset alert flag after playing sound

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Skip iteration if the camera isn't ready

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Read trackbar values
    blink_threshold_float = blink_threshold / 100.0  # Direct 20-30 scale to 0.20-0.30
    alert_delay = alert_delay_slider  # Directly use the slider value (0-30 seconds)

    # Default status
    face_detected = "No"
    blink_detected = "No"
    ear_display = "N/A"  # Default when no face is found

    if faces:  # Only process if a face is detected
        face_detected = "Yes"
        
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[LEFT_EYE]
            right_eye = shape[RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0  # Average EAR
            ear_display = f"{ear:.3f}"  # Format EAR for display

            if ear < blink_threshold_float:
                if blink_detected == "No":  # Log only on state change
                    blink_detected = "Yes"
                blink_time = time.time()

            # Check if no blink happened for alert_delay seconds
            elif time.time() - blink_time > alert_delay and not alert_active:
                alert_active = True
                threading.Thread(target=play_alert).start()
                # Draw eyes on frame
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

    else:
        # No face detected, reset blink timer to prevent false alerts
        blink_time = time.time()

    # Display status text on preview
    # Left side of frame
    cv2.putText(frame, f"Face Detected: {face_detected}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blink Detected: {blink_detected}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Right side of frame
    cv2.putText(frame, f"Trigger: {blink_threshold}", (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Alert Delay: {alert_delay} sec", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Count Down Seconds Until Alert Triggers
    remaining_time = int(alert_delay - (time.time() - blink_time))
    cv2.putText(frame, f"Alert in {remaining_time} sec", (400, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    frame_height = frame.shape[0]
    cv2.putText(frame, "H: Help | Q: Quit", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if show_help:
        cv2.putText(frame, f"Current Value: {ear_display}", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        help_text = [
            "Blinky Help",
            "--------------------------------",
            "Trigger: Adjusts blink sensitivity", 
            "*Note: 20 = less sensitive, 30 = more sensitive",
            "Alert Delay: Time before an alert if no blinks",
            " ",
            " ",
            "Press 'Q' to Quit",
            "Press 'H' to Hide Help"
        ]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        y0, dy = frame.shape[0] // 3, 30
        for i, line in enumerate(help_text):
            y = y0 + i * dy
            cv2.putText(frame, line, (frame.shape[1] // 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Blinky Eye Blink Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('h'):  # Toggle help screen
        show_help = not show_help

cap.release()
cv2.destroyAllWindows()
