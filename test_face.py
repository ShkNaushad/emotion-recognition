import cv2
import os

cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
print('Cascade path:', cascade_path)
print('File exists:', os.path.exists(cascade_path))

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise SystemExit('ERROR: face detector load nahi hua')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit('ERROR: webcam nahi khuli')

print('Webcam chalu. Band karne ke liye video window pe q dabao.')

while True:
    ok, frame = cap.read()
    if not ok:
        print('ERROR: frame nahi mila')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()