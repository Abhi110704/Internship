import cv2
import numpy as np
import time

def recognize_and_mark_attendance():
    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read('attendance_model.yml')
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Face cascade could not be loaded.")
        return

    # Define the label mapping
    label_map = {
        0: "Student1",
        1: "Student2",
        2: "Student3"
    }

    attendance = {}  # To store marked attendance
    cam = cv2.VideoCapture(0)  # Open the webcam

    if not cam.isOpened():
        print("Error: Camera could not be opened.")
        return

    frame_skip = 5  # Process every 5th frame
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Process every N-th frame
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]

            # Predict the face
            try:
                label, confidence = recognizer.predict(face_image)
            except Exception as e:
                print(f"Error in prediction: {e}")
                continue

            # Stricter confidence threshold
            if confidence < 50:  # Lower confidence means better match
                student_name = label_map.get(label, "Unknown")
                if student_name != "Unknown" and student_name not in attendance:
                    attendance[student_name] = True  # Mark attendance
                    print(f"Marked attendance for: {student_name}")

                # Display on the frame
                text = f"{student_name} ({int(confidence)}%)"
                cv2.putText(resized_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(resized_frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw a rectangle around the face
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the frame
        cv2.imshow('Mark Attendance', resized_frame)

        # Quit on pressing 'q' or after 30 seconds
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 30):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Display attendance summary
    if attendance:
        print("\nAttendance Marked:")
        for student in attendance.keys():
            print(student)
    else:
        print("No attendance marked.")

if __name__ == "__main__":
    recognize_and_mark_attendance()
