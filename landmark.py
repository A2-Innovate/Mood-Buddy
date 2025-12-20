import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def extract_features(landmarks):
    mar = get_distance(landmarks[13], landmarks[14]) / (get_distance(landmarks[78], landmarks[308]) + 1e-6)
    ear = get_distance(landmarks[159], landmarks[145]) / (get_distance(landmarks[33], landmarks[133]) + 1e-6)
    brow = get_distance(landmarks[70], landmarks[159])
    
    ref_x, ref_y = landmarks[1].x, landmarks[1].y
    dist = get_distance(landmarks[33], landmarks[263]) + 1e-6
    
    row = [mar, ear, brow] 
    for p in landmarks:
        row.extend([(p.x - ref_x)/dist, (p.y - ref_y)/dist]) 
    return row

def process_dataset(folder_path, output_name):
    data_rows = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    emotions = os.listdir(folder_path)
    for emotion in emotions:
        emotion_dir = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_dir): continue
        images = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.jpg', '.png'))]
        for img_name in tqdm(images, desc=f"Processing {emotion}"):
            img = cv2.imread(os.path.join(emotion_dir, img_name))
            if img is None: continue
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                features = extract_features(results.multi_face_landmarks[0].landmark)
                features.append(emotion)
                data_rows.append(features)
    
    if data_rows:
        pd.DataFrame(data_rows).to_csv(output_name, index=False)
        print(f"Saved: {output_name}")
    else:
        print(f"No data found in {folder_path}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, "dataset", "train")
test_path = os.path.join(BASE_DIR, "dataset", "test")

process_dataset(train_path, "face_train.csv")
process_dataset(test_path, "face_test.csv")