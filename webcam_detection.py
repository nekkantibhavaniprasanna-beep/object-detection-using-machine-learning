import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    counts = {}

    for box in results[0].boxes:

        class_id = int(box.cls)

        object_name = results[0].names[class_id]

        counts[object_name] = counts.get(object_name, 0) + 1

    y = 30

    for obj, count in counts.items():

        text = f"{obj}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 30

    if counts.get("person", 0) >= 3:

        cv2.putText(
            annotated_frame,
            "Crowd Alert!",
            (10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()