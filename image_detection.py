from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("images/test.jpg", save=True)

counts = {}

for box in results[0].boxes:
    class_id = int(box.cls)
    name = results[0].names[class_id]
    counts[name] = counts.get(name, 0) + 1

print("\nDetected Objects (Image):")

total = 0

for obj, count in counts.items():
    print(f"{obj}: {count}")
    total += count

print("\nTotal Objects:", total)

# ALERTS
if counts.get("person", 0) >= 3:
    print("⚠ Crowd Alert (Image)")