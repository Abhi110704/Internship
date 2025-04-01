import cv2
import os

def collect_faces(student_name):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)  # Use webcam

    dataset_path = 'dataset'
    student_folder = os.path.join(dataset_path, student_name)

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    print(f"Collecting images for {student_name}. Press 'q' to stop.")
    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            count += 1
            face_image = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{student_folder}/{student_name}_{count}.jpg", face_image)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Collecting Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Collected {count} images for {student_name}.")

student_name = input("Enter student's name or ID: ")
collect_faces(student_name)
