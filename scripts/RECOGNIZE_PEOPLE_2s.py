#!/usr/bin/env python3

"""people_recognizer.py

# KNN + embedding cloud za prepoznavo oseb
Naloži model ki ga je naredil build_embeddings.py.

recognizer = PeopleRecognizer("./embeddings_db")
result = recognizer.recognize(frame)   # frame = cv2 numpy array ali pot do slike
# result = {"name": "Luka", "gender": "male", "job": "accountant", "confidence": 0.87}
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from deepface import DeepFace


DEEPFACE_MODEL = "Facenet"
UNKNOWN_MAX_COSINE_DISTANCE = 0.40  # 0.0 = enak, 1.0 = povsem drugačen


class PeopleRecognizer:
    def __init__(
        self,
        db_dir: str = "./embeddings_db",
        unknown_max_distance: float = UNKNOWN_MAX_COSINE_DISTANCE,
    ):
        db = Path(db_dir)
        self.unknown_max_distance = float(unknown_max_distance)

        meta_path = db / "metadata.json"
        model_path = db / "knn_model.pkl"

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}")

        with open(meta_path) as f:
            self.metadata = json.load(f)

        with open(model_path, "rb") as f:
            saved = pickle.load(f)
            self.knn = saved["knn"]
            self.le = saved["label_encoder"]

        self._deepface = DeepFace

    def _get_embedding(self, image) -> np.ndarray | None:
        try:
            result = self._deepface.represent(
                img_path=image,
                model_name=DEEPFACE_MODEL,
                enforce_detection=False,
                detector_backend="opencv",
            )
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"Embedding napaka: {e}")
            return None

    def recognize(self, image) -> dict | None:
        emb = self._get_embedding(image)
        if emb is None:
            return None

        emb_2d = emb.reshape(1, -1)

        # nearest neighbor distance (cosine)
        min_dist = float(self.knn.kneighbors(emb_2d, n_neighbors=1)[0][0][0])
        if min_dist > self.unknown_max_distance:
            return None

        # predict class + confidence
        confidence = None
        best_name = None

        try:
            proba = self.knn.predict_proba(emb_2d)[0]
            best_ix = int(np.argmax(proba))
            class_id = int(self.knn.classes_[best_ix])
            best_name = self.le.inverse_transform([class_id])[0]
            confidence = float(proba[best_ix])
        except Exception:
            # fallback if predict_proba isn't available
            class_id = int(self.knn.predict(emb_2d)[0])
            best_name = self.le.inverse_transform([class_id])[0]
            confidence = float(max(0.0, 1.0 - min_dist))

        if not best_name:
            return None

        # gender (best-effort)
        gender = "?"
        try:
            analyze_result = DeepFace.analyze(img_path=image, actions=["gender"], enforce_detection=False)
            gender = analyze_result[0].get("dominant_gender")
        except Exception:
            return None

        meta = self.metadata.get(best_name, {})

        return {
            "name": best_name,
            "gender": gender,
            "job": meta.get("job", "unknown"),
            "confidence": round(confidence, 3),
            "distance": round(min_dist, 3),
        }


# try on one picture
# ros2 run pkg recognize_people_2s -- <image_path>"
def main():
    parser = argparse.ArgumentParser(description="Run face recognition on a single image.")
    parser.add_argument("image", nargs="?", help="Path to an image file")
    parser.add_argument("--db", default="./embeddings_db", help="Folder created by build_embeddings.py")
    parser.add_argument("--threshold", type=float, default=UNKNOWN_MAX_COSINE_DISTANCE, help="Max cosine distance")
    args = parser.parse_args()

    if not args.image:
        print("Usage: ros2 run pkg recognize_people_2s -- <image_path>")
        return 0

    recognizer = PeopleRecognizer(args.db, unknown_max_distance=args.threshold)
    result = recognizer.recognize(args.image)
    if result is None:
        print("No face recognized.")
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())