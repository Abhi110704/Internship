import cv2
import numpy as np
import os

def train_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    dataset_path = 'dataset'

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            label_map[current_label] = folder
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(current_label)
            current_label += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save('attendance_model.yml')
    print("Model trained and saved as attendance_model.yml.")
    print("Label mapping:", label_map)

train_face_recognizer()
