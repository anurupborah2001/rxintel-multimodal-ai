from ultralytics import YOLO

model = YOLO("yolov12n.pt")  # Load pretrained weights
results = model.train(
    data="datasets/receipt_dataset_v5/data.yaml",
    epochs=200,  # Increase epochs (CPU is slow, but you need more iterations on small data)
    batch=4,  # Lower batch size, CPU RAM-friendly
    imgsz=640,  # Keep standard YOLO size
    device="cpu",  # Force CPU
    patience=50,  # Early stopping if no improvement
    lr0=0.005,  # Lower learning rate (tiny dataset)
    optimizer="Adam",  # Adam is more stable on small datasets
    # Augmentation
    scale=0.5,  # Random scaling
    mosaic=0.7,  # Some mosaic, but not too much
    mixup=0.0,  # Avoid mixup (confuses small dataset)
    copy_paste=0.2,  # Can help with synthetic variation
    hsv_h=0.015,  # Slight hue variation
    hsv_s=0.7,  # Saturation variation
    hsv_v=0.4,  # Brightness variation
    flipud=0.2,  # Vertical flip (20%)
    fliplr=0.5,  # Horizontal flip (50%)
    project="train-runs",
    name="v5",
    # Keeping it for reference
    # epochs=100,  # Recommended minimum for fine-tuning
    # batch=16,    # Adjust based on GPU memory
    # optimizer="Adam",
    # #  imgsz=640,   # Image size
    # device="cpu",  # GPU device (use "cpu" if no GPU)
    # scale=0.5,   # Data augmentation: scale
    # mosaic=1.0,  # Mosaic augmentation
    # mixup=0.0,   # Mixup augmentation
    # copy_paste=0.1, # Copy-paste augmentation.
    # project="train-runs",
    # name="v5"
)
print(results)
# Using CLI we can trian as below:
# yolo train model=yolov12n.pt data=datasets/yolov12/receipts_dataset/data.yaml epochs=50 imgsz=640 batch=16 device=cpu project=pharmacy-models name=v2
