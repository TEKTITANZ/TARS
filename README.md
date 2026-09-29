# ISRO-Astrotrack: Edge Operator Console

ISRO-Astrotrack is an offline, computer-vision-based tracking system designed to assist astronauts with deep-space payload experiments. Built for strict operational constraints, the system provides a zero-latency, cloud-independent graphical dashboard paired with a robust object-tracking AI engine to monitor, verify, and guide users through sequential physical tasks.

## Core Features

*   **100% Offline Operation:** Runs entirely on edge hardware without cloud connectivity, ensuring reliability in deep-space environments.
*   **Asynchronous Multi-Threading:** Separates heavy AI vision processing (`VisionThread`) from the main CustomTkinter GUI thread, maintaining a fluid UI and zero-latency HD webcam feed.
*   **Zero-Latency Audio Engine:** Implements a threaded, sequential `.wav` playback queue with cooldowns to prevent voice overlapping and audio desync during rapid experiment steps.
*   **Robust Object Tracking Pipeline:** 
    *   **YOLOv8:** Semantic detection of the main container, red box, and yellow box.
    *   **Kalman Filter:** 8-state constant-velocity tracking stabilizes the main container's bounding box even during partial occlusion.
    *   **MediaPipe:** Skeletal and wrist tracking for precise human-object interaction and lid-opening gesture recognition.
*   **Strict Anti-Hallucination Filtering:** Mathematically rejects false positives by enforcing strict confidence thresholds (85%+), minimum pixel sizes, aspect-ratio bounds, and HSV color-density checks.
*   **Anti-Speedrun Constraints:** Requires sustained, deliberate physical actions (e.g., 15 frames/0.5 seconds of sustained contact or displacement) before validating a step to prevent accidental triggers.

## Tech Stack

*   **Language:** Python 3.10+
*   **GUI Framework:** `CustomTkinter`, `PIL` (Pillow)
*   **Computer Vision:** `OpenCV` (`cv2`)
*   **AI Models:** `ultralytics` (YOLOv8), `mediapipe` (Pose tracking)
*   **Audio:** `winsound` (Windows native), `pyttsx3` (Fallback/Dynamic TTS)
*   **Math/Logic:** `numpy`, `threading`, `queue`

## Project Structure

```text
ISRO-Astrotrack/
│
├── dashboard.py         # Main entry point. Handles CustomTkinter GUI, threaded video rendering, and state logs.
├── tracker.py           # Core AI engine. Houses YOLO, MediaPipe, Kalman filter, and step-verification logic.
├── best.pt              # Trained YOLOv8 model weights (must be in root directory).
├── anomaly.wav          # Zero-latency audio alert for incorrect actions.
├── success_0.wav        # Zero-latency audio cue for Step 1 completion.
├── success_1.wav        # Zero-latency audio cue for Step 2 completion.
├── ...                  # Additional success step audio files.
└── README.md            # Project documentation.


Experiment Sequence Supported
The engine is pre-configured to track and verify the following 7-step sequence:

Touch the main box.

Open the main box (vertical wrist travel).

Unload the yellow box.

Unload the red box.

Load the red box.

Load the yellow box.

Close the main box.

Installation & Setup
Clone the repository to your local edge device.

Install dependencies using pip:

pip install customtkinter opencv-python numpy ultralytics mediapipe pyttsx3 pillow

Verify Assets: Ensure best.pt and all pre-recorded .wav files (anomaly.wav, success_0.wav, etc.) are located in the same directory as the Python scripts.

Hardware Verification: Ensure a USB or integrated webcam is connected. The system defaults to camera_index=1 for external webcams (adjustable in dashboard.py if using an integrated laptop camera on index=0).

Usage
Launch the dashboard by running the main interface script:
python dashboard.py

The system will automatically initialize the webcam, load the YOLO and MediaPipe models into memory, and begin tracking the workspace. Ensure the physical workspace is well-lit to assist the HSV color-density tracking loops.
