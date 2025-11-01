import face_recognition
import cv2
import pickle
import os

# --- Settings ---
WEBCAM_INDEX = 0  # Default webcam is usually 0
NAME = "your_name"  # Change this to your username
ENCODING_FILE = f"{NAME}_encoding.pkl"
# ----------------

print("Starting video capture...")
# Initialize webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open video device at index {WEBCAM_INDEX}")
    exit()

print("Please look directly at the camera. Press 's' to save your face.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Video - Press "s" to save', frame)

    # Wait for the 's' key to be pressed
    if cv2.waitKey(1) & 0xFF == ord("s"):
        print("Capturing image...")

        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the image
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 0:
            print("No face detected. Please try again.")
        elif len(face_locations) > 1:
            print("Multiple faces detected. Please ensure only you are in the frame.")
        else:
            print("Face detected. Creating encoding...")
            # Get the face encoding
            known_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[
                0
            ]

            # Save the encoding to a file
            data = {"name": NAME, "encoding": known_encoding}
            with open(ENCODING_FILE, "wb") as f:
                pickle.dump(data, f)

            print(f"Success! Your face encoding has been saved to {ENCODING_FILE}")
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
