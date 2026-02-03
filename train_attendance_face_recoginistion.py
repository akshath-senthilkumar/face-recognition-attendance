import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# Initialize MobileFaceNet (lightweight)
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

DATASET_PATH = '/content/drive/MyDrive/labeled_faces'

X_embeddings = []
y_labels = []

for person in tqdm(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    for img_file in os.listdir(person_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (320, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img)
        if not faces:
            continue

        X_embeddings.append(faces[0].embedding)
        y_labels.append(person)

X_embeddings = np.array(X_embeddings)
y_labels = np.array(y_labels)

np.save("cosine_embeddings.npy", X_embeddings)
np.save("cosine_labels.npy", y_labels)

print(f"Stored {len(X_embeddings)} embeddings from {len(np.unique(y_labels))} people.")
