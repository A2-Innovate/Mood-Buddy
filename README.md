# Mood Buddy

🚀 **[Live Demo](https://mood-buddy.onrender.com/)** | 💻 **[GitHub Repository](https://github.com/A2-Innovate/Mood-Buddy)**

Mood Buddy is a web-based application that performs real-time facial emotion recognition and delivers context-aware, persona-driven responses using Generative AI.

The system captures webcam feeds, extracts geometric facial landmarks, and classifies the user's current emotion using a custom-trained machine learning model. Based on the detected mood and a user-selected persona (e.g., Professional, Strict Mentor, Friend), it interfaces with the Gemini API to generate tailored advice.

## Tech Stack

  * **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js (for session analytics)
  * **Backend:** Python, Flask, Flask-CORS
  * **Computer Vision:** OpenCV, Google MediaPipe (Face Mesh)
  * **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
  * **Generative AI:** Google Gemini API (Gemini 2.5 Flash)

## Key Features

  * **Real-Time Feature Extraction:** Utilizes MediaPipe Face Mesh to extract 468 facial landmarks. Calculates dynamic spatial features including Mouth Aspect Ratio (MAR), Eye Aspect Ratio (EAR), and eyebrow distances for robust emotion detection.
  * **Custom Emotion Classifier:** Employs an XGBoost classifier trained on custom tabular landmark data, prioritizing inference speed and accuracy over heavier deep learning approaches.
  * **Context-Aware AI Generation:** Integrates the Gemini API to generate dynamic text responses. The system uses strict prompt engineering to maintain distinct personas (Friend, Career Counselor, Roaster, Strict Mentor, Poet) while keeping latency low.
  * **Session Analytics:** Tracks and visualizes emotion distribution over the current session using Chart.js.

## System Architecture

1.  **Client:** Captures a snapshot from the local webcam via the browser's MediaDevices API and sends it as a blob to the backend.
2.  **Processing:** Flask receives the image, decodes it via OpenCV, and passes it to the ML pipeline.
3.  **Inference:** MediaPipe extracts landmarks; custom math functions normalize and compute geometric ratios. The XGBoost model predicts the emotion.
4.  **Generation:** The predicted emotion and user-selected role are passed to the Gemini API wrapper, which returns a formatted response.
5.  **Delivery:** The prediction, confidence score, and AI response are returned to the client and rendered on the dashboard.

## Data Set
```bash
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
```

## Installation and Setup

### Prerequisites

  * Python 3.8+
  * A valid Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/A2-Innovate/Mood-Buddy.git
cd Mood-Buddy
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.

## Model Training

The emotion classification model was built to be lightweight. If you wish to retrain the model with your own data:

1.  Ensure your training and testing datasets (`face_train.csv`, `face_test.csv`) are in the root directory. The data should consist of normalized facial landmark coordinates and geometric ratios.
2.  Run the training script:

<!-- end list -->

```bash
python train.py
```

This will output the test accuracy and generate `buddy_xgb_model.pkl` and `label_encoder.pkl` for the inference pipeline.

## Project Structure

```text
├── app.py                 # Flask server and API routing
├── main.py                # CV pipeline, feature extraction, and Gemini integration
├── train.py               # XGBoost model training script
├── requirements.txt       # Project dependencies
├── index.html             # Main application interface
├── style.css              # Custom styling
├── script.js              # Frontend logic and state management
├── buddy_xgb_model.pkl    # Pre-trained XGBoost model
└── label_encoder.pkl      # Pre-trained Label Encoder
```
