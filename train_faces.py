import face_recognition
import pickle
import os

def train_faces():
    known_faces_dir = 'known_faces'
    known_encodings = []
    known_names = []

    for student_dir in os.listdir(known_faces_dir):
        student_path = os.path.join(known_faces_dir, student_dir)
        if os.path.isdir(student_path):
            for image_file in os.listdir(student_path):
                image_path = os.path.join(student_path, image_file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(student_dir)

    data = {"encodings": known_encodings, "names": known_names}
    with open("face_encodings.pickle", "wb") as f:
        pickle.dump(data, f)

    print("Huấn luyện hoàn tất! Số học sinh:", len(set(known_names)))

if __name__ == "__main__":
       train_faces()