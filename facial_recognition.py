import face_recognition
import cv2
import numpy as np
import time
import pickle
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

# ESP32-CAM configuration
ESP32_CAM_IP = "192.168.137.137"
ESP32_UNLOCK_URL = f"http://{ESP32_CAM_IP}:8080/door"
ESP32_LOCK_URL = f"http://{ESP32_CAM_IP}:8080/door"
ESP32_CAM_URL = f'http://{ESP32_CAM_IP}:81/stream'

# Email configuration
SENDER_EMAIL = "daocongson10012004@gmail.com"
SENDER_PASSWORD = "yjnq ggps ohci bfga"
RECIPIENT_EMAIL = "lekhauhuutai48@gmail.com"  # Replace with your actual email

# Threshold configuration for face recognition
RECOGNITION_THRESHOLD = 0.6  # Adjust between 0.5-0.7 for optimal results
FACE_DISTANCE_THRESHOLD = RECOGNITION_THRESHOLD
CONFIDENCE_THRESHOLD = 0.8

# Load pre-trained face encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

cv_scaler = 4
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Status tracking
last_status = "locked"
last_unlock_time = 0
UNLOCK_DURATION = 7  # Seconds to keep door unlocked
currentname = "unknown"

# Unknown face tracking
unknown_start_time = None  # Time when unknown face was first detected
unknown_duration_threshold = 5  # Seconds threshold to send alert email
last_email_time = 0
EMAIL_COOLDOWN = 60  # Seconds between emails to prevent spamming

def send_email_with_image(image_path, recipient_email, subject="Cảnh báo: Phát hiện người lạ!"):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Create email
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Email body
        body = "Hệ thống đã phát hiện người lạ liên tục trong hơn 5 giây. Vui lòng kiểm tra hình ảnh đính kèm."
        msg.attach(MIMEText(body, 'plain'))

        # Attach image
        if os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(image_path)}",
            )
            msg.attach(part)

        # Connect to SMTP server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"Email đã được gửi tới {recipient_email} thành công.")
    except Exception as e:
        print(f"Không thể gửi email: {e}")

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
    global face_locations, face_encodings, face_names, currentname, unknown_start_time, last_email_time

    # Resize for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Use HOG model for faster detection
    face_locations = face_recognition.face_locations(rgb_resized_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    detected_known = False
    unknown_detected = False

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
            unknown_start_time = None  # Reset unknown timer if known face detected
        else:
            display_name = "Unknown"
            unknown_detected = True

        face_names.append(display_name)

    # Handle unknown face tracking and email alerts
    current_time = time.time()
    if unknown_detected and not detected_known:
        if unknown_start_time is None:
            unknown_start_time = current_time  # Set start time
        else:
            elapsed_time = current_time - unknown_start_time
            # Check if unknown face has been detected continuously and email cooldown period passed
            if (elapsed_time >= unknown_duration_threshold and
                (current_time - last_email_time) > EMAIL_COOLDOWN):
                print("[ALERT] Unknown person detected for over 5 seconds, sending alert email...")
                unknown_image_path = "unknown_detected.jpg"
                cv2.imwrite(unknown_image_path, frame)  # Save image of unknown person
                send_email_with_image(unknown_image_path, RECIPIENT_EMAIL)
                last_email_time = current_time  # Update the last email time
                unknown_start_time = current_time  # Reset the timer but keep tracking
    elif not unknown_detected:
        unknown_start_time = None  # Reset if no unknown faces

    # Door control based on face detection
    if detected_known and last_status == "locked":
        print("[INFO] Known face detected, unlocking...")
        send_unlock()
    elif not detected_known and last_status == "unlocked":
        if current_time - last_unlock_time > UNLOCK_DURATION:
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
    print(f"[INFO] Email alerts enabled for unknown faces (threshold: {unknown_duration_threshold}s)")

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

        # Display unknown detection timer if active
        if unknown_start_time is not None:
            unknown_time = time.time() - unknown_start_time
            cv2.putText(display_frame, f"Unknown detected: {unknown_time:.1f}s",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Facial Recognition', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    print("[INFO] Facial recognition terminated")