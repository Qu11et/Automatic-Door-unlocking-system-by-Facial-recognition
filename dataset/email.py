import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import time

# Hàm gửi email
def send_email_with_image(image_path, recipient_email, subject="Cảnh báo: Phát hiện người lạ!"):
    sender_email = "daocongson10012004@gmail.com"  # Thay bằng email của bạn
    sender_password = "yjnq ggps ohci bfga"  # Thay bằng mật khẩu ứng dụng Gmail
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        # Tạo email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Nội dung email
        body = "Hệ thống đã phát hiện người lạ liên tục trong hơn 10 giây. Vui lòng kiểm tra hình ảnh đính kèm."
        msg.attach(MIMEText(body, 'plain'))

        # Đính kèm hình ảnh
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

        # Kết nối đến SMTP server và gửi email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        print(f"Email đã được gửi tới {recipient_email} thành công.")
    except Exception as e:
        print(f"Không thể gửi email: {e}")

# Biến theo dõi thời gian phát hiện người lạ
unknown_start_time = None  # Thời gian bắt đầu phát hiện người lạ
unknown_duration_threshold = 10  # Ngưỡng thời gian (giây) để gửi email

def process_frame(frame):
    global face_locations, face_encodings, face_names, currentname, unknown_start_time

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
            unknown_start_time = None  # Reset thời gian nếu phát hiện người quen
        else:
            display_name = "Unknown"

            # Xử lý thời gian phát hiện người lạ
            if unknown_start_time is None:
                unknown_start_time = time.time()  # Đặt thời gian bắt đầu
            else:
                elapsed_time = time.time() - unknown_start_time
                if elapsed_time >= unknown_duration_threshold:
                    print("[ALERT] Phát hiện người lạ liên tục quá 10 giây, gửi email cảnh báo...")
                    unknown_image_path = "unknown.jpg"
                    cv2.imwrite(unknown_image_path, frame)  # Lưu hình ảnh người lạ
                    send_email_with_image(unknown_image_path, "owner_email@gmail.com")
                    unknown_start_time = None  # Reset thời gian sau khi gửi email

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