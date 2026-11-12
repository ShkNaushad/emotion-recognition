import os
import time
import json
import csv
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- Model aur face detector load ----------
model = tf.keras.models.load_model('model/emotion_model.keras')
class_names = json.load(open('model/class_names.json'))

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise SystemExit('ERROR: face detector load nahi hua')

os.makedirs('outputs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ---------- Session ka CSV log ----------
log_path = datetime.now().strftime('logs/session_%Y%m%d_%H%M%S.csv')
log_file = open(log_path, 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['time_sec', 'dominant'] + class_names)
session_start = time.time()
last_log = 0.0

# Har emotion ka apna rang (BGR)
COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 140, 0),
    'fear': (180, 0, 180),
    'happy': (0, 220, 255),
    'neutral': (200, 200, 200),
    'sad': (255, 120, 0),
    'surprise': (0, 165, 255),
}

PANEL_W = 300


def predict_emotion(gray_face):
    face = cv2.resize(gray_face, (48, 48))
    x = face.astype('float32').reshape(1, 48, 48, 1)  # model ke andar hi /255 hota hai
    probs = model(x, training=False).numpy()[0]
    return probs


def draw_panel(height, probs, fps):
    panel = np.full((height, PANEL_W, 3), 30, dtype=np.uint8)
    cv2.putText(panel, 'EMOTIONS', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f'FPS: {fps:.1f}', (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

    if probs is None:
        cv2.putText(panel, 'No face found', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
    else:
        top = int(np.argmax(probs))
        for i, name in enumerate(class_names):
            y = 70 + i * 48
            pct = float(probs[i]) * 100
            color = COLORS.get(name, (255, 255, 255))
            thick = 2 if i == top else 1
            cv2.putText(panel, f'{name}', (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick)
            cv2.putText(panel, f'{pct:4.1f}%', (225, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick)
            cv2.rectangle(panel, (15, y + 8), (15 + 270, y + 20), (70, 70, 70), -1)
            cv2.rectangle(panel, (15, y + 8), (15 + int(270 * probs[i]), y + 20), color, -1)

    cv2.putText(panel, 'q = quit   s = screenshot', (15, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return panel


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit('ERROR: webcam nahi khuli')

print('App chalu. q = band karo, s = screenshot save karo (outputs folder mein).')

smooth = None       # pichle frames ka average, taaki bars jhatke se na hilein
prev_time = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print('ERROR: frame nahi mila')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    main_probs = None
    if len(faces) > 0:
        # sabse bade chehre ke liye bars dikhayenge
        main_idx = int(np.argmax([w * h for (_, _, w, h) in faces]))
        for idx, (x, y, w, h) in enumerate(faces):
            probs = predict_emotion(gray[y:y + h, x:x + w])
            if idx == main_idx:
                smooth = probs if smooth is None else 0.6 * smooth + 0.4 * probs
                main_probs = smooth
                label_probs = smooth
            else:
                label_probs = probs
            top = int(np.argmax(label_probs))
            name = class_names[top]
            color = COLORS.get(name, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{name} {label_probs[top] * 100:.0f}%', (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        smooth = None

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    # Har 0.5 second mein ek row CSV mein
    if main_probs is not None and now - last_log >= 0.5:
        t = now - session_start
        dom = class_names[int(np.argmax(main_probs))]
        log_writer.writerow([round(t, 2), dom] + [round(float(p), 4) for p in main_probs])
        last_log = now

    panel = draw_panel(frame.shape[0], main_probs, fps)
    combined = np.hstack([frame, panel])
    cv2.imshow('Facial Emotion Recognition', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        fname = time.strftime('outputs/emotion_%Y%m%d_%H%M%S.png')
        cv2.imwrite(fname, combined)
        print('Screenshot saved:', fname)

cap.release()
log_file.close()
cv2.destroyAllWindows()
print('Log saved:', log_path)