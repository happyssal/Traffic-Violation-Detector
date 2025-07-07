Traffic Violation Detector
Project Description
This project is an AI-powered computer vision system designed to detect traffic light violations from video footage. It leverages the YOLO (You Only Look Once) model for real-time object detection of cars and red traffic lights. Users can interactively define a virtual "stop line" within the video frame. The system then tracks vehicles and identifies violations when a car crosses this user-defined stop line while a red traffic light is present and within proximity. The output is a processed video, visually annotated with bounding boxes around detected objects and alerts for any identified violations.

Features
Object Detection: Utilizes a custom-trained YOLO model to identify cars and red traffic lights.

Interactive ROI Selection: Allows users to graphically define a two-point line segment as the "stop line" within the first frame of the video.

Vehicle Tracking: Tracks individual vehicles across frames to monitor their movement relative to the stop line.

Red Light Violation Detection: Automatically detects instances where a car crosses the defined stop line when a red traffic light is active and nearby.

Annotated Output: Generates a new video file with bounding boxes, object IDs, confidence scores, stop line visualization, and violation alerts.

Configurable Thresholds: Easily adjustable confidence thresholds and proximity parameters directly within the script.

Demonstration
Here's an example of the system in action, showing detected objects, the defined stop line, and a violation:

Installation
To set up and run this project, follow these steps:

1. Clone the Repository
First, clone this GitHub repository to your local machine:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME # Navigate into your project directory


(Replace YOUR_USERNAME and YOUR_REPOSITORY_NAME with your actual GitHub username and repository name.)

2. Create and Activate a Virtual Environment
It's highly recommended to use a Python virtual environment to manage dependencies.

python -m venv venv
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux (Bash/Zsh):
source venv/bin/activate


3. Install Dependencies
With your virtual environment activated, install the required Python packages:

pip install ultralytics opencv-python numpy


Note: During ultralytics installation, you might see warnings about missing dependencies like lap. ultralytics often attempts to auto-install these. If lap installation fails due to file access errors (common on Windows), try to manually install it after restarting your terminal/VS Code: pip install lap==0.5.12 or pip install lap>=0.5.12.

4. Place Model Weights
This project relies on a pre-trained YOLO model. Ensure your best.pt model file is located at the specified path within your project structure:

YOUR_REPOSITORY_NAME/
├── runs/
│   └── detect/
│       └── traffic_violation_detector6/
│           └── weights/
│               └── best.pt  <-- Place your model file here
├── detect_violations.py
├── ... (other project files)


If you trained your model, place your best.pt file there. If you obtained it from elsewhere, ensure it's in this exact path or update the MODEL_PATH variable in detect_violations.py.

Usage
To run the traffic violation detector:

Activate your virtual environment (if not already active):

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux (Bash/Zsh):
source venv/bin/activate


Run the script:

python detect_violations.py


Provide Video Path: The script will prompt you to enter the full path to your input video file (e.g., C:/Users/YourName/video.mp4 or /home/user/videos/traffic.mp4).

Select Stop Line ROI: A window will appear showing the first frame of your video.

Click two points on the image to define your stop line.

After selecting two points, press the 'c' key to confirm your selection.

Press 'r' to reset your selection.

Press 'q' to quit the application.

Processing: The script will then start processing the video, applying detection and tracking, and checking for violations.

Output: A new video file named [original_video_name]_processed.mp4 will be saved in the output_videos/ directory within your project folder.

Configuration
You can adjust various parameters by modifying the constants at the beginning of the detect_violations.py file:

MODEL_PATH: Path to your YOLO model weights.

CONF_THRESHOLD: Confidence threshold for car detection.

RED_LIGHT_CONF_THRESHOLD: Confidence threshold for red traffic light detection.

RED_LIGHT_PROXIMITY_THRESHOLD: Maximum distance a car can be from a red light to be considered for a violation.

MIN_DETECTION_FRAMES: Minimum consecutive frames an object must be detected to be considered a stable track.

Project Structure (Key Files)
TrafficViolationDetector/
├── detect_violations.py      # Main script for detection and violation logic
├── runs/                     # Directory for Ultralytics runs (e.g., model weights)
│   └── detect/
│       └── traffic_violation_detector6/
│           └── weights/
│               └── best.pt   # Your YOLO model file
├── output_videos/            # Directory where processed videos will be saved
├── .gitignore                # Specifies files and folders to ignore by Git
├── README.md                 # This file
├── image_8a87ea.jpg          # Example image for demonstration
└── venv/                     # Python Virtual Environment (ignored by Git)


Contributing
Contributions are welcome! If you have suggestions for improvements or find issues, please feel free to open an issue or submit a pull request.
