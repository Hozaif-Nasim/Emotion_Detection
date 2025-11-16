import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from collections import deque
import time
import os

# ---------- Load model ----------
with open("emotiondetector.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("emotiondetection.h5")

# ---------- Labels (must match training order you gave) ----------
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# ---------- Smoothing (majority vote over last N predictions) ----------
SMOOTH_WINDOW = 15
history = deque(maxlen=SMOOTH_WINDOW)

# ---------- Face detector (Haarcascade) ----------
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# ---------- helper: preprocess face to model input ----------
def preprocess_face(face_gray, target_size=(48,48), padding=20):
    # add some padding around the face region (helps match training crops)
    h, w = face_gray.shape
    # resize before padding to avoid very large images when padding over edges
    face = cv2.resize(face_gray, target_size)
    # histogram equalization to normalize contrast (makes webcam resemble dataset)
    face = cv2.equalizeHist(face)
    # reshape for model (batch, h, w, 1) and normalize
    face = face.reshape(1, target_size[0], target_size[1], 1).astype('float32') / 255.0
    return face

# ---------- Start webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera access.")

# Optional: lower resolution for speed (uncomment if needed)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # compute FPS
    cur_time = time.time()
    fps = 0.9*fps + 0.1*(1.0 / (cur_time - prev_time + 1e-8))
    prev_time = cur_time

    # convert to gray and equalize histogram for lighting invariance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # detect faces (returns x,y,w,h). scaleFactor and minNeighbors tuned for webcam
    faces = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    # For each face, predict and display
    for (x, y, w, h) in faces:
        # expand bbox slightly (clamp to image bounds)
        pad = int(0.15 * max(w, h))  # 15% padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_roi = gray_eq[y1:y2, x1:x2]
        try:
            face_input = preprocess_face(face_roi, target_size=(48,48))
        except Exception:
            continue

        # predict (single forward pass)
        preds = model.predict(face_input, verbose=0)
        pred_class = int(np.argmax(preds))

        # push to history and compute majority vote
        history.append(pred_class)
        final_class = max(set(history), key=history.count)
        emotion_text = labels[final_class].capitalize()

        # draw bounding box and emotion text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    # press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
