import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/test.mp4")

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    results = model(frame)

    annotated = results[0].plot()

    counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls)
        name = results[0].names[class_id]
        counts[name] = counts.get(name, 0) + 1

    # ALERT LOGIC
    if counts.get("person", 0) >= 5:
        cv2.putText(annotated, "CROWD ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

    cv2.imshow("Video Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()