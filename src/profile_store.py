"""Person profile management with face encoding persistence.

Stores person profiles (name, preferences, relationships) and LBPH
face recognizer model for person identification.

OpenCV is an optional dependency. Without it, face recognition is disabled.
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    cv2 = None
    np = None


@dataclass
class PersonProfile:
    """A known person: identity, preferences, and face training data."""
    name: str
    face_id: int = 0
    preferences: str = ""
    relationship: str = ""
    notes: str = ""
    registered_at: float = 0.0
    last_seen: float = 0.0
    sample_count: int = 0


class ProfileStore:
    """Face recognition model + person profile persistence.

    Uses OpenCV LBPHFaceRecognizer. Gracefully degrades when cv2 is absent.
    """

    def __init__(self, data_dir: str = "data/profiles"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.profiles: dict[int, PersonProfile] = {}
        self._name_index: dict[str, int] = {}
        self._trained = False
        self._recognizer = None
        self._face_cascade = None

        if HAVE_CV2:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade)

        self._load()

    # ── public API ────────────────────────────────────────────

    def register_person(self, name: str, face_frames: list,
                        preferences: str = "",
                        relationship: str = "") -> Optional[PersonProfile]:
        """Register or update a person with face samples for training."""
        if not HAVE_CV2:
            logger.warning("[Profiles] cv2 not available, skipping registration")
            return None

        if name in self._name_index:
            face_id = self._name_index[name]
            profile = self.profiles[face_id]
            profile.preferences = preferences or profile.preferences
            profile.relationship = relationship or profile.relationship
        else:
            face_id = max(self.profiles.keys(), default=0) + 1
            profile = PersonProfile(
                name=name, face_id=face_id,
                preferences=preferences, relationship=relationship,
                registered_at=time.time(),
            )
            self.profiles[face_id] = profile
            self._name_index[name] = face_id

        faces, labels = self._collect_training_data()
        person_dir = self.data_dir / str(face_id)
        person_dir.mkdir(parents=True, exist_ok=True)
        existing_count = len(list(person_dir.glob("*.png")))
        for i, frame in enumerate(face_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            faces.append(gray)
            labels.append(face_id)
            cv2.imwrite(str(person_dir / f"{existing_count + i}.png"), gray)

        profile.sample_count += len(face_frames)
        profile.last_seen = time.time()

        if faces:
            self._recognizer.train(faces, np.array(labels))
            self._trained = True
            logger.info(f"[Profiles] Trained recognizer for '{name}' "
                        f"(face_id={face_id}, samples={profile.sample_count})")

        self._save()
        return profile

    def recognize(self, face_frame) -> Optional[PersonProfile]:
        """Identify a face crop. Returns profile or None."""
        if not HAVE_CV2 or not self._trained:
            return None

        gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        face_id, confidence = self._recognizer.predict(gray)
        if confidence < 0:
            return None

        threshold = config.vision.recognition_threshold
        if confidence > threshold:
            return None

        profile = self.profiles.get(face_id)
        if profile:
            profile.last_seen = time.time()
            logger.info(f"[Profiles] Recognized '{profile.name}' (conf={confidence:.0f})")
            self._save()
        return profile

    def detect_faces(self, frame) -> list:
        """Detect all faces in a frame. Returns list of face crops (BGR)."""
        if not HAVE_CV2 or self._face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(config.vision.min_face_size, config.vision.min_face_size),
        )
        return [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]

    def detect_and_recognize(self, frame) -> list[dict]:
        """Detect + recognize. Returns [{name, bbox, profile}, ...]."""
        if not HAVE_CV2:
            return []

        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(config.vision.min_face_size, config.vision.min_face_size),
        )
        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            profile = self.recognize(crop)
            results.append({
                "bbox": (int(x), int(y), int(w), int(h)),
                "crop": crop, "profile": profile,
                "name": profile.name if profile else "unknown",
            })
        return results

    def get_profile(self, name: str) -> Optional[PersonProfile]:
        fid = self._name_index.get(name)
        return self.profiles.get(fid) if fid else None

    def get_all_profiles(self) -> list:
        return list(self.profiles.values())

    def save_profile(self, name: str, relationship: str = "",
                     preferences: str = "") -> PersonProfile:
        """Create or update a person's profile without face training."""
        if name in self._name_index:
            face_id = self._name_index[name]
            profile = self.profiles[face_id]
        else:
            face_id = max(self.profiles.keys(), default=0) + 1
            profile = PersonProfile(
                name=name, face_id=face_id,
                relationship=relationship, preferences=preferences,
                registered_at=time.time(),
            )
            self.profiles[face_id] = profile
            self._name_index[name] = face_id
        profile.relationship = relationship
        profile.preferences = preferences
        profile.last_seen = time.time()
        self._save()
        logger.info(f"[Profiles] Saved profile for '{name}'")
        return profile

    def remove_person(self, name: str) -> bool:
        fid = self._name_index.pop(name, None)
        if fid and fid in self.profiles:
            del self.profiles[fid]
            self._retrain_all()
            self._save()
            return True
        return False

    # ── internals ──────────────────────────────────────────────

    def _collect_training_data(self):
        faces, labels = [], []
        if not HAVE_CV2:
            return faces, labels
        for face_id in self.profiles:
            sp = self.data_dir / str(face_id)
            if sp.exists():
                for img_path in sorted(sp.glob("*.png")):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(face_id)
        return faces, labels

    def _retrain_all(self):
        if not HAVE_CV2:
            return
        faces, labels = self._collect_training_data()
        if faces:
            self._recognizer.train(faces, np.array(labels))
            self._trained = True

    def _save(self):
        if self._trained and HAVE_CV2:
            self._recognizer.save(str(self.data_dir / "lbph_model.yml"))
        meta = {str(k): asdict(v) for k, v in self.profiles.items()}
        with open(self.data_dir / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load(self):
        if HAVE_CV2:
            mp = self.data_dir / "lbph_model.yml"
            if mp.exists():
                self._recognizer.read(str(mp))
                self._trained = True
                logger.info(f"[Profiles] Loaded recognizer from {mp}")

        mp = self.data_dir / "profiles.json"
        if mp.exists():
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for fid_str, data in meta.items():
                p = PersonProfile(**data)
                fid = int(fid_str)
                self.profiles[fid] = p
                self._name_index[p.name] = fid
            logger.info(f"[Profiles] Loaded {len(self.profiles)} profiles")


from .config import config  # noqa: E402

profile_store = ProfileStore()
