from flask import Flask, Response, render_template
import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import threading
import os
import time

app = Flask(__name__)

with open("face_encodings.pickle", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]


camera = cv2.VideoCapture(0)  
if not camera.isOpened():
    print("Lỗi: Không thể mở webcam với CAP_DEFAULT!")
 
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not camera.isOpened():
        print("Lỗi: Không thể mở webcam với CAP_MSMF!")
        exit()
print("Webcam đã mở thành công!")
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

attendance_list = []
attendance_set = set()
tolerance = 0.5
max_distance = 0.5
lock = threading.Lock()

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Lỗi: Không đọc được frame từ webcam!")
            time.sleep(1) 
            continue

       
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2


            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            name = "Unknown"

            if distance <= max_distance:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                if matches[best_match_index]:
                    name = known_names[best_match_index]


            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({distance:.3f})" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


            if name != "Unknown" and name not in attendance_set:
                with lock:
                    attendance_set.add(name)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_list.append({"name": name, "timestamp": timestamp, "distance": distance})
                    with open("output\\attendance.csv", "a") as f:
                        f.write(f"{name},{timestamp},{distance:.3f}\n")


        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Lỗi: Không thể mã hóa frame thành JPEG!")
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  

@app.route('/')
def index():
    return render_template('index.html', attendance_list=attendance_list)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        camera.release()
        cv2.destroyAllWindows()