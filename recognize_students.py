import face_recognition
import cv2
import pickle
import numpy as np
from datetime import datetime

def recognize_faces():
    # Load dữ liệu huấn luyện
    try:
        with open("face_encodings.pickle", "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file face_encodings.pickle. Vui lòng chạy train_faces.py trước.")
        return

    # Khởi tạo camera
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        print("Lỗi: Không thể truy cập webcam!")
        return

    attendance = set()
    tolerance = 0.5  # Giảm tolerance để tăng độ chính xác
    max_distance = 0.5  # Ngưỡng distance tối đa nghiêm ngặt hơn

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam!")
            break

        # Resize frame để tăng tốc
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect và encode khuôn mặt
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Tính khoảng cách
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            name = "Unknown"

            # Chỉ lấy match tốt nhất nếu thỏa mãn ngưỡng
            if distance <= max_distance:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    # Debug: In tất cả các match gần ngưỡng
                    print(f"Debug: Match tốt nhất: {name} (distance: {distance:.3f})")
                    for i, (d, match) in enumerate(zip(face_distances, matches)):
                        if match and d <= max_distance:
                            print(f"Debug: Match khác: {known_names[i]} (distance: {d:.3f})")

            # Vẽ hộp và tên
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({distance:.3f})" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Ghi nhận điểm danh
            if name != "Unknown" and name not in attendance:
                attendance.add(name)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{name} đã điểm danh lúc {timestamp} (distance: {distance:.3f})")
                with open("output\\attendance.csv", "a") as f:
                    f.write(f"{name},{timestamp},{distance:.3f}\n")

        cv2.imshow('Nhận Dạng Khuôn Mặt', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("Danh sách điểm danh:", attendance)

if __name__ == "__main__":
    recognize_faces()