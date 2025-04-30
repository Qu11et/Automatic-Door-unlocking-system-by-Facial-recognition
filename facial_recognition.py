import face_recognition
import cv2
import numpy as np
import time
import pickle
import requests

# Địa chỉ ESP32-CAM (thay bằng IP thật)
ESP32_CAM_IP = "192.168.1.13"
ESP32_UNLOCK_URL = f"http://{ESP32_CAM_IP}:8080/door"
ESP32_LOCK_URL = f"http://{ESP32_CAM_IP}:8080/door"

# Threshold configuration for face recognition
RECOGNITION_THRESHOLD = 0.6  # Adjust between 0.5-0.7 for optimal results

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# URL stream của ESP32 CAM (thường là cổng 81)
ESP32_CAM_URL = f'http://{ESP32_CAM_IP}:81/stream'

cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

FACE_DISTANCE_THRESHOLD = RECOGNITION_THRESHOLD
CONFIDENCE_THRESHOLD = 0.8

# Trạng thái nhận diện trước đó để tránh gửi POST liên tục
last_status = "locked"
last_unlock_time = 0
UNLOCK_DURATION = 7  # Giây mở cửa
currentname = "unknown"

def send_unlock():
    global last_status, last_unlock_time
    try:
        r = requests.post(ESP32_UNLOCK_URL, data="unlock", timeout=2)
        print("Unlock signal sent:", r.status_code)
        last_status = "unlocked"
        last_unlock_time = time.time()
    except Exception as e:
        print("Failed to send unlock:", e)

def send_lock():
    global last_status
    try:
        r = requests.post(ESP32_LOCK_URL, data="lock", timeout=2)
        print("Lock signal sent:", r.status_code)
        last_status = "locked"
    except Exception as e:
        print("Failed to send lock:", e)

def process_frame(frame):
    global face_locations, face_encodings, face_names, currentname

    # Resize for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Use HOG model for faster detection
    face_locations = face_recognition.face_locations(rgb_resized_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    detected_known = False

    for encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)

        # Find the best match (smallest distance)
        best_match_index = None
        min_distance = float('inf')

        for i, face_distance in enumerate(face_distances):
            if face_distance < min_distance:
                min_distance = face_distance
                best_match_index = i

        confidence = 1.0 - min_distance if min_distance < 1.0 else 0.0

        if best_match_index is not None and confidence >= RECOGNITION_THRESHOLD:
            name = known_face_names[best_match_index]
            display_name = f"{name} ({confidence:.2%})"
            detected_known = True

            if currentname != name:
                currentname = name
                print(f"[INFO] Recognized: {name} with confidence {confidence:.2%}")
        else:
            display_name = "Unknown"

        face_names.append(display_name)

    # Gửi tín hiệu mở/đóng khóa nếu trạng thái thay đổi
    if detected_known and last_status == "locked":
        print("[INFO] Known face detected, unlocking...")
        send_unlock()
    elif not detected_known and last_status == "unlocked":
        if time.time() - last_unlock_time > UNLOCK_DURATION:
            print("[INFO] No known face, locking...")
            send_lock()

    return frame

def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        if "Unknown" in name:
            box_color = (0, 0, 255)  # Red for unknown
            status_text = "ACCESS DENIED"
        else:
            box_color = (0, 255, 0)  # Green for known
            status_text = "ACCESS GRANTED"

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, status_text, (left, y), font, 0.6, box_color, 2)
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

def mjpeg_stream(url):
    import requests
    stream = requests.get(url, stream=True)
    bytes_data = b''
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield frame

try:
    print("[INFO] Starting facial recognition system...")
    print(f"[INFO] Recognition threshold: {RECOGNITION_THRESHOLD}")
    print(f"[INFO] ESP32 camera URL: {ESP32_CAM_URL}")

    for frame in mjpeg_stream(ESP32_CAM_URL):
        if frame is None:
            print("Cannot receive frame from video stream!")
            break

        processed_frame = process_frame(frame)
        display_frame = draw_results(processed_frame)

        current_fps = calculate_fps()
        cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Recognition Threshold: {RECOGNITION_THRESHOLD}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        lock_status = f"Door: {last_status.upper()}"
        if last_status == "unlocked":
            time_left = max(0, UNLOCK_DURATION - (time.time() - last_unlock_time))
            lock_status += f" ({time_left:.1f}s)"
        cv2.putText(display_frame, lock_status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow('Facial Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    print("[INFO] Facial recognition terminated")