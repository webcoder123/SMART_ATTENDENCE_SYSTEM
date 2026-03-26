import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

print("[INFO] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

dataset_path = "dataset"

known_embeddings = []
known_names = []

print("[INFO] Encoding faces...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)
        faces = app.get(image)

        if len(faces) > 0:
            embedding = faces[0].embedding
            known_embeddings.append(embedding)
            known_names.append(person_name)

data = {
    "embeddings": known_embeddings,
    "names": known_names
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding complete.")