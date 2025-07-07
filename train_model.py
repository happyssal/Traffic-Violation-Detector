import os
from ultralytics import YOLO

def train_yolov8_model(data_yaml_path, epochs=100, img_size=640, batch_size=16, project_name='traffic_violation_detector', model_name='yolov8n.pt'):
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=project_name,
        cache=True
    )
    print(f"Training completed. Results saved in runs/detect/{project_name}/")
    return results

if __name__ == "__main__":
    DATA_YAML = "data.yaml"
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 8
    print(f"Starting YOLOv8 training with {EPOCHS} epochs...")
    train_yolov8_model(
        data_yaml_path=DATA_YAML,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        project_name='traffic_violation_detector',
        model_name='yolov8n.pt'
    )
    print("Training script finished.")