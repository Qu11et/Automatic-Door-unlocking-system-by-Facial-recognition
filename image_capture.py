import cv2
import os
from datetime import datetime

ESP32_CAM_URL = 'http://192.168.137.88:81/stream'

PERSON_NAME = "Khau"


def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder


def capture_photos(name):
    folder = create_folder(name)

    video_capture = cv2.VideoCapture(ESP32_CAM_URL)

    if not video_capture.isOpened():
        print("Cannot open video stream!")
        return

    photo_count = 0

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Cannot receive frame from video stream!")
            break

        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    video_capture.release()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")


if __name__ == "__main__":
    capture_photos(PERSON_NAME)