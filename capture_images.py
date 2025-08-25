import cv2
import os

def capture_images():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW cho Windows
    count = 0
    student_id = input("Nhập ID_Tên (ví dụ: 001_NguyenVanA): ")
    os.makedirs(f"known_faces\\{student_id}", exist_ok=True)
    while count < 10:  # Chụp 10 ảnh
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập webcam!")
            break
        cv2.imshow("Chụp ảnh", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Nhấn 's' để lưu
            save_path = f"known_faces\\{student_id}\\image_{count}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"Đã lưu ảnh tại: {save_path}")
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()