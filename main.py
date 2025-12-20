import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import google.generativeai as genai
from dotenv import load_dotenv

class AI_Assistant:
    def __init__(self):
        self.model = None
        self.is_configured = False

    def configure(self, api_key_env_var="GEMINI_API_KEY", manual_key=""):
        api_key = manual_key
        if "TERI_ASLI" in api_key or not api_key:
            load_dotenv()
            api_key = os.getenv(api_key_env_var)

        if not api_key:
            print("❌ ERROR: API Key nahi mili!")
            return False

        try:
            genai.configure(api_key=api_key)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash", 
                generation_config={"temperature": 0.7, "max_output_tokens": 1100},
                safety_settings=safety_settings
            )
            self.is_configured = True
            print("✅ Gemini AI (2.5 Flash) Configured Successfully.")
            return True
        except Exception as e:
            print(f"❌ Config Error: {e}")
            return False

    def get_advice(self, emotion, role="Friend"):
        """
        🔮 UPDATED: No Extra Spaces + Short & Crisp Responses
        """
        if not self.is_configured:
            return "AI not connected."

        personas = {
            "Friend": (
                f"You are a supportive college project partner. User is feeling {emotion}. "
                "Speak in decent Hinglish. Be sensible but warm. "
                "Give advice like 'Thoda break lele' or 'Talk to someone'. "
                f"End with a Song Recommendation matching {emotion} mood. "
                "STRICT INSTRUCTION: Keep the response short and concise. Do NOT add unnecessary empty lines or huge gaps."
            ),
            
            "Professional": (
                f"You are a calm Indian Career Counselor. User is feeling {emotion}. "
                "Speak in professional English with a warm tone. "
                "Focus on mental balance and productivity. "
                "End with a calm instrumental/lofi track name. "
                "STRICT INSTRUCTION: Keep the text compact. No extra line breaks between sentences."
            ),
            
            "Roaster": (
                f"You are a witty and sarcastic friend. User is feeling {emotion}. "
                "Tease them lightly about their expression using smart humor. "
                "End by sarcastically suggesting a song that mocks their mood. "
                "STRICT INSTRUCTION: Keep it punchy and short. Do not waste token space with empty lines."
            ),
            
            "Strict": (
                f"You are a focused Mentor. User is feeling {emotion}. "
                "Be firm. Push them to get back to work immediately. "
                "End with a High-Energy Song recommendation. "
                "STRICT INSTRUCTION: Be direct and to the point. No filler text or extra spaces."
            ),
            
            "Poet": (
                f"You are a Poet. User is feeling {emotion}. "
                "Write strictly only 2 lines of Shayari in Roman Hindi. "
                "Do NOT add any explanation or context. "
                "Immediately after the Shayari, just add: 'Bas yaad rakh, mast rehne ka!'. "
                "STRICT INSTRUCTION: Do not add any extra newlines. Keep the format very tight."
            )
        }

        selected_prompt = personas.get(role, personas["Friend"])
        
        try:
            response = self.model.generate_content(selected_prompt)
            if response.text:
                clean_text = response.text.strip().replace("*", "").replace("\n\n\n", "\n")
                return clean_text
            else:
                return "Thinking..."
        except Exception as e:
            if "429" in str(e): return "Quota Limit! Wait a bit."
            print(f"API Error: {e}")
            return "Just breathe. Everything will be okay."

def load_ml_models(model_path, encoder_path):
    try:
        model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        print("✅ ML Models Loaded.")
        return model, le
    except FileNotFoundError:
        print("❌ ERROR: .pkl files missing.")
        return None, None

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_features_live(face_landmarks):
    landmarks = face_landmarks.landmark
    mar = get_distance(landmarks[13], landmarks[14]) / (get_distance(landmarks[78], landmarks[308]) + 1e-6)
    ear = get_distance(landmarks[159], landmarks[145]) / (get_distance(landmarks[33], landmarks[133]) + 1e-6)
    brow = get_distance(landmarks[70], landmarks[159])
    ref_x, ref_y = landmarks[1].x, landmarks[1].y
    dist = get_distance(landmarks[33], landmarks[263]) + 1e-6
    row = [mar, ear, brow]
    for p in landmarks:
        if (len(row) - 3) // 2 >= 468: break
        row.extend([(p.x - ref_x) / dist, (p.y - ref_y) / dist])
    return np.array(row).reshape(1, -1)

api_model = None
api_le = None
api_face_mesh = None

def load_resources_globally():
    global api_model, api_le, api_face_mesh
    if api_model is None:
        print("📥 Loading models for API...")
        api_model, api_le = load_ml_models("buddy_xgb_model.pkl", "label_encoder.pkl")
        mp_face = mp.solutions.face_mesh
        api_face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def predict_mood(image):
    try:
        load_resources_globally()
        if api_model is None: return "Model Error", 0.0
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = api_face_mesh.process(rgb)
        
        if not results.multi_face_landmarks: return "No Face Detected", 0.0
        
        features = extract_features_live(results.multi_face_landmarks[0])
        
        if features.shape[1] > api_model.n_features_in_: 
            features = features[:, :api_model.n_features_in_]
            
        pred = api_model.predict(features)[0]
        mood = api_le.inverse_transform([pred])[0]
        
        confidence = 0.0
        if hasattr(api_model, "predict_proba"):
            probs = api_model.predict_proba(features)[0]
            confidence = float(np.max(probs)) * 100 
        else: 
            confidence = 100.0 
            
        return mood, round(confidence, 2)
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return "Error", 0.0