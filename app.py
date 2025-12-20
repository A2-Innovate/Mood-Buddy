from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os

from main import predict_mood, AI_Assistant

app = Flask(__name__, static_url_path='', static_folder='.', template_folder='.')
CORS(app)

bot = AI_Assistant()
bot.configure() 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
            
        file = request.files['image']
        npimg = np.fromfile(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        mood, confidence = predict_mood(img)
        final_suggestion = "Stay cool."
                
        if mood not in ["Error", "No Face Detected"]:
            try:
                user_role = request.form.get('role', 'Friend')
                
                final_suggestion = bot.get_advice(mood, user_role)

            except Exception as e:
                print(f"AI Error: {e}")
                final_suggestion = get_suggestion(mood)
        else:
            final_suggestion = "Could not detect face properly."

        return jsonify({
            "mood": mood,
            "confidence": confidence,
            "suggestion": final_suggestion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_suggestion(mood):
    suggestions = {
        "Happy": "Great energy! Keep crushing it!",
        "Sad": "Take a deep breath. Listen to some Lo-Fi.",
        "Angry": "Count to 10. Maybe drink some water?",
        "Neutral": "Focus mode on. Let's get work done."
    }
    return suggestions.get(mood, "Stay cool.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)