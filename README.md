# Blinky Eye Blink Detector

A lightweight Python application that detects eye blinks using OpenCV and dlib, with adjustable sensitivity and alert delay settings.

## Features
- Auto-detects the correct camera (skips non-working or virtual cameras)
- Displays real-time EAR (Eye Aspect Ratio) values
- Adjustable blink sensitivity and alert delay (values saved between runs)
- Plays a warning beep if no blink is detected for too long
- Toggleable help menu
- Simple GUI with OpenCV

## Requirements
- Python 3.8+  
- Required pip packages (install via `pip install -r requirements.txt`):
    ```sh
    pip install opencv-python numpy scipy matlibplot dlib imutils pyinstaller
    ```

- Download Required Model:  
    You must download `shape_predictor_68_face_landmarks.dat` from:  
    [shape_predictor_68_face_landmarks.dat](https://github.com/GuoQuanhao/68_points/blob/master/shape_predictor_68_face_landmarks.dat)  
    Place it in the same directory as `blinky.py`.

### **Dlib Installation Note**
On Windows, `dlib` may require installation from an **elevated command prompt (Run as Administrator)**. 
If you encounter issues, you may need cmake and visual studio build tools installed.
Or maybe install a precompiled version from [https://pypi.org/project/dlib/](https://pypi.org/project/dlib/).

## How to Build as an Executable
To package this as a standalone `.exe` (Windows):
```sh
pyinstaller --onefile --noconsole --add-data "shape_predictor_68_face_landmarks.dat;." blinky.py
```
This will generate `dist/blinky.exe`, which can be run without Python installed.

## Usage
1. Run `blinky.py` or `blinky.exe`
2. Adjust sensitivity with the **"Trigger"** slider (higher = more sensitive)
3. Adjust alert delay with the **"Delay(s)"** slider (higher = longer wait before alert)
4. Press **H** for help, **Q** to quit

## Notes
- If the camera selection takes a few seconds, it is auto-detecting the first valid camera. Virtual cameras can cause this to take longer.
- The application stores slider settings in `blinkconfig.json` for convenience on subsequent runs.
- If packaging with PyInstaller, ensure the `.dat` file is included (see command above).

## License
This project is licensed under the MIT License.

## Author
**Nick Rotundo**  
[GitHub](https://github.com/nickjrotundo/blinky)
