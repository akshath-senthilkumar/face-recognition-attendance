!pip install insightface onnxruntime scikit-learn opencv-python pytz

import cv2
import numpy as np
import joblib
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import drive
from datetime import datetime, timedelta
import pytz
import csv

# Mount Drive
drive.mount('/content/drive')

# Load known embeddings and labels
known_embeddings = np.load("cosine_embeddings.npy")
known_labels = np.load("cosine_labels.npy")

# Load face model
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Paths
video_path = '/content/drive/MyDrive/Person14.mp4'
output_path = '/content/drive/MyDrive/output_IST_timestamp.mp4'
csv_path = '/content/drive/MyDrive/timestamp_log_IST.csv'

# Video setup
cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Timezone for IST
ist = pytz.timezone('Asia/Kolkata')

# Trackers
logged_known = set()
unknown_start = None
unknown_logged = False
threshold = 0.6

# CSV Init
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Timestamp (IST)'])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_ist_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        has_unknown = False
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding.reshape(1, -1)

            sims = cosine_similarity(emb, known_embeddings)[0]
            best_score = np.max(sims)
            best_match = np.argmax(sims)

            if best_score >= threshold:
                name = known_labels[best_match]
                label = f"{name} ({best_score:.2f})"

                # Log only once
                if name not in logged_known:
                    writer.writerow([name, current_ist_time])
                    logged_known.add(name)
            else:
                name = "Unknown"
                label = name
                has_unknown = True

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Handle unknown timer
        if has_unknown:
            if unknown_start is None:
                unknown_start = datetime.now()
            elif not unknown_logged and (datetime.now() - unknown_start).total_seconds() >= 5:
                writer.writerow(["Unknown (⚠️ >5s presence)", current_ist_time])
                unknown_logged = True
        else:
            unknown_start = None
            unknown_logged = False

        out.write(frame)

cap.release()
out.release()

print(f"✅ Annotated video saved: {output_path}")
print(f"📝 Timestamp log saved: {csv_path}")
