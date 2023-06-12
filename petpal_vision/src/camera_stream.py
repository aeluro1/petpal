from ultralytics import YOLO
import cv2 as cv

# model = YOLO("yolov8n.pt")

# results = model.predict(stream = True, )

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

while True:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not retrieve frame from buffer")
    cv.imshow("cam", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()