#!/usr/bin/env python
import face_recognition
import cv2
import pickle
import sys
import os

# --- Settings ---
WEBCAM_INDEX = 0
# ENCODING_FILE = "/lib/security/my_face_auth/your_name_encoding.pkl"  # Use the file from Step 1
ENCODING_FILE = "/opt/my_face_auth/your_name_encoding.pkl"  # Use the file from Step 1
TIMEOUT_SEC = 5  # How long to try before failing
# ----------------


def log(message):
    """Simple logger for PAM script"""
    with open("/var/log/my_face_auth.log", "a") as f:
        f.write(f"[{os.getpid()}] {message}\n")


def load_known_face():
    """Loads the saved face encoding from disk"""
    try:
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
            return data["encoding"]
    except FileNotFoundError:
        log(f"Error: Encoding file not found at {ENCODING_FILE}")
        return None


def check_face():
    known_encoding = load_known_face()
    if known_encoding is None:
        return False

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        log(f"Error: Could not open video device at index {WEBCAM_INDEX}")
        return False

    log("Starting face check...")

    # We use a simple loop instead of a hard timeout for simplicity
    for _ in range(TIMEOUT_SEC * 10):  # Try for ~5 seconds (10 frames/sec)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        if len(face_locations) > 0:
            # We only care about the first face found
            current_face_encoding = face_recognition.face_encodings(
                rgb_frame, face_locations
            )[0]

            # Compare the face to the known encoding
            matches = face_recognition.compare_faces(
                [known_encoding], current_face_encoding, tolerance=0.55
            )

            if matches[0]:
                log("SUCCESS: Face match found.")
                cap.release()
                return True

        # Small delay
        cv2.waitKey(100)

    log("FAILURE: No face match found in time.")
    cap.release()
    return False


if __name__ == "__main__":
    if check_face():
        # Exit with 0 for success
        sys.exit(0)
    else:
        # Exit with 1 for failure
        sys.exit(1)
