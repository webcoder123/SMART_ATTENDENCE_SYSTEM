import cv2
import pickle
import numpy as np
from datetime import datetime
import csv
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

print("[INFO] Loading embeddings...")
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = np.array(data["embeddings"])
known_names = data["names"]

print("[INFO] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

marked_today = set()

def mark_attendance(name):
    if name in marked_today:
        return

    file_exists = os.path.isfile("attendance.csv")

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()
        writer.writerow([name,
                         now.strftime("%Y-%m-%d"),
                         now.strftime("%H:%M:%S")])

    marked_today.add(name)
    print(f"[INFO] Attendance marked for {name}")


cap = cv2.VideoCapture(0)

print("[INFO] Starting camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        embedding = face.embedding.reshape(1, -1)

        similarities = cosine_similarity(embedding, known_embeddings)
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]

        name = "Unknown"

        if best_score > 0.5:  # threshold (0.4–0.6 recommended)
            name = known_names[best_match_index]
            mark_attendance(name)

        bbox = face.bbox.astype(int)
        cv2.rectangle(frame,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      (0, 255, 0),
                      2)

        cv2.putText(frame,
                    f"{name} ({best_score:.2f})",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()