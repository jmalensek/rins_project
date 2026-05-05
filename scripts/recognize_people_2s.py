import json
import numpy as np
from deepface import DeepFace

MODEL = "Facenet"
THRESHOLD = 0.40  # Cosine similarity prag (0.0 = enak, 1.0 = povsem drugačen)

class PeopleRecognizer:
    def __init__(self, embeddings_file="embeddings.json"):
        with open(embeddings_file, "r") as f:
            data = json.load(f)
        
        self.names     = [d["name"]      for d in data]
        self.genders   = [d["gender"]    for d in data]
        self.jobs      = [d["job"]       for d in data]
        self.embeddings = np.array([d["embedding"] for d in data])
        
        print(f"✓ Naloženih {len(self.names)} oseb iz baze")

    def _cosine_distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize(self, image):
        """
        image: pot do slike (str) ali numpy array (OpenCV frame)
        Vrne: dict z name, gender, job, confidence – ali None če ni zaznave
        """
        try:
            result = DeepFace.represent(
                img_path=image,
                model_name=MODEL,
                enforce_detection=True
            )
            query_embedding = np.array(result[0]["embedding"])
        except Exception as e:
            print(f"Ni mogoče zaznati obraza: {e}")
            return None

        # Cosine distance do vseh v bazi
        distances = np.array([
            self._cosine_distance(query_embedding, emb)
            for emb in self.embeddings
        ])
        
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        
        if best_dist > THRESHOLD:
            return {"name": "Unknown", "gender": "?", "job": "?", "confidence": 0.0}
        
        return {
            "name":       self.names[best_idx],
            "gender":     self.genders[best_idx],
            "job":        self.jobs[best_idx],
            "confidence": round(1 - best_dist, 3)
        }