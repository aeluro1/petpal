from ultralytics import YOLO
import cv2 as cv

MODEL_PATH = "/path/to/best.pt"
model = YOLO(MODEL_PATH)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

while True:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not retrieve frame from buffer")

    frame = model.predict(source = frame, show = True, save = True, save_txt = True)

    cv.imshow("cam", frame)
    
    if cv.waitKey(1) == ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()