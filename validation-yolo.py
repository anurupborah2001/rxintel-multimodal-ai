# Alternative CLI
# yolo val model=pharmacy-models/v1/weights/best.pt data=datasets/yolov12/receipts_dataset/data.yaml

from ultralytics import YOLO

model = YOLO("train-runs/v5/weights/best.pt")  # Load the best weights
metrics = model.val(data="datasets/receipt_dataset_v5/data.yaml")
print(metrics.box.map)  # mAP@50:95
print(metrics.box.map50)  # mAP@50
# Access the class mapping
class_map = model.names

print("Class ID → Class Name mapping:")
for class_id, class_name in class_map.items():
    print(f"  {class_id}: {class_name}")
