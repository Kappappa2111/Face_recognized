import cv2
from mtcnn import MTCNN

detector = MTCNN()

def mtcnn_face_detect(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame)

    face = []
    if faces:
        confidence_scores = [face['confidence'] for face in faces]
        max_confidence_index = confidence_scores.index(max(confidence_scores))
        face = faces[max_confidence_index]['box']
    return face

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face = mtcnn_face_detect(frame)
    if face:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
