ISRO-Astrotrack: Edge Operator Console
ISRO-Astrotrack is an offline, computer-vision-based tracking system designed to assist astronauts with deep-space payload experiments. Built for strict operational constraints, the system provides a zero-latency, cloud-independent graphical dashboard paired with a robust object-tracking AI engine to monitor, verify, and guide users through sequential physical tasks.

Live demo: https://drive.google.com/drive/folders/1IE7YjwG73IVBqYPSzY9etL6TPkzOVUNt?usp=sharing

SIH Details
Team Name: TEKTITANZ

Team ID: 163149

Problem Statement ID: SIH26174

Problem Statement Title: AI Human Activity Recognition for On-board BAS Experiments

Problem Statement
During deep-space missions, astronauts must follow complex, sequential operational protocols for payload experiments. Currently, ensuring strict adherence to these multi-step physical procedures requires constant manual oversight or post-experiment analysis, which is prone to human error and difficult to scale. There is a critical need for a localized, offline system capable of autonomously tracking objects, interpreting human-object interactions in real-time, and providing zero-latency auditory and visual feedback without relying on external cloud connectivity.

Solution
This platform acts as an autonomous edge operator console. It utilizes a connected HD webcam to process physical workspace interactions entirely offline. The system leverages a multi-threaded architecture to decouple heavy AI vision processing from the graphical user interface. By applying a custom-trained YOLOv8 model combined with a Kalman Filter and MediaPipe, the system accurately tracks the main experiment container and payload boxes. It enforces strict procedural compliance, sequentially guiding the astronaut through unload/load steps with zero-latency audio cues while mathematically rejecting false-positive bounding box "hallucinations."

Key Features
100% Offline Edge Operation: Runs entirely on local hardware without internet connectivity.

Asynchronous Multi-Threading: Separates AI vision processing from the CustomTkinter GUI for a fluid, zero-latency interface.

Sequential Audio Engine: A dedicated, strict audio queue prevents voice overlapping and desyncing.

Robust Object Tracking: YOLOv8 semantic detection paired with an 8-state Kalman Filter for object occlusion handling.

Anti-Hallucination Mathematics: Rejects false positive detections through strict confidence thresholds (85%+), minimum pixel bounding, aspect-ratio filtering, and HSV color-density verification.

Anti-Speedrun Constraints: Requires sustained physical interaction (e.g., 15 continuous frames / 0.5 seconds) to validate experimental steps and prevent accidental triggers.

Actionable Activity Log: Real-time mission elapsed time (MET), automated status updates, and anomaly flagging.

System Architecture
The system is a decoupled offline application: a heavy VisionThread backend handles frame-by-frame AI inference and mathematical filtering, pushing validated states to a fast GUI thread.

TARS:https://drive.google.com/drive/folders/1IE7YjwG73IVBqYPSzY9etL6TPkzOVUNt?usp=sharing


The pipeline runs in continuous loops. First, OpenCV reads the hardware webcam buffer. Second, YOLOv8 and MediaPipe process the frame to extract bounding boxes and wrist landmarks. Third, custom logic filters out anomalies based on aspect ratios, size, and color density. Fourth, human-to-object distances and sustained touch counters evaluate if an experimental step (e.g., "unload the yellow box") is fulfilled. Finally, the state manager updates the graphical dashboard and triggers zero-latency .wav files and dynamic TTS for astronaut guidance.

Tech Stack
Frontend/GUI: CustomTkinter with Pillow (PIL) for modern, responsive desktop rendering.

Backend/Vision: OpenCV (cv2) for hardware camera interfacing and matrix operations.

AI and Inference: Ultralytics (YOLOv8) for object detection, MediaPipe for human pose/wrist tracking.

Audio/Feedback: winsound for zero-latency deterministic wav playback, pyttsx3 for dynamic text-to-speech fallback.

State Management: Python threading, queue, and numpy for asynchronous data handling and distance mathematics.

Project Structure
Plaintext
├── dashboard.py         # Main entry point. Handles CustomTkinter GUI, threaded video rendering, and state logs.
├── tracker.py           # Core AI engine. Houses YOLO, MediaPipe, Kalman filter, and step-verification logic.
├── best.pt              # Trained YOLOv8 model weights (must be in root directory).
├── anomaly.wav          # Zero-latency audio alert for incorrect actions.
├── success_0.wav        # Zero-latency audio cue for Step 1 completion.
├── success_1.wav        # Zero-latency audio cue for Step 2 completion.
├── ...                  # Additional success step audio files.
├── diagrams/
│   └── architecture.png # System architecture workflow diagram
└── screenshots/
    ├── dashboard.png    # Live Operator Console UI screenshot
    └── timeline.png     # Experiment progress and activity log screenshot
Installation & Setup
You'll need Python 3.10 or later. No cloud API keys or database connections are required, as the system is fully isolated for edge computing.

From the project root:

Bash
git clone  https://github.com/TEKTITANZ/TARS/blob/main/TARS
cd SIH_PROJECT-SIH26174

# Create a virtual environment (Optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install required dependencies
pip install customtkinter opencv-python numpy ultralytics mediapipe pyttsx3 pillow
Verify Assets: Ensure best.pt and all pre-recorded .wav files (anomaly.wav, success_0.wav, etc.) are located in the same directory as the Python scripts.

Environment Variables
No environment variables are required.

The system relies entirely on local hardware constraints. Camera inputs are hardcoded securely in dashboard.py (e.g., camera_index=0 for integrated webcams, camera_index=1 for external webcams).

How to Run
Ensure your physical workspace is well-lit for accurate HSV color-density tracking, and your webcam is connected.

Start the dashboard from the root folder:

Bash
python dashboard.py
The system will automatically initialize the webcam, load the YOLO and MediaPipe models into memory, announce the system startup via audio, and begin tracking the workspace.

Screenshots / Demo
Dashboard - Edge Operator Console

The main operator dashboard showing the live HD webcam feed with tracking overlays, hardware status, and the zero-latency audio wave visualizer.

History View / Progress Log

The live experiment progression ledger, showcasing the chronological completion of steps, MET timer, and dynamically flagged AI anomalies.

Deployment
The system is deployed directly onto edge hardware (Windows/Linux local machines or hardened laptops) intended for field or space operations. It does not require hosting services like Vercel or Render.

Future Improvements
Custom dynamic sequence loading via local JSON files to support varying daily experiment checklists.

Enhanced 3D gesture recognition for more complex interaction requirements (e.g., twisting, pushing, or precise tool manipulation).

Automated end-of-mission PDF report generation summarizing MET timelines and anomaly occurrences.
