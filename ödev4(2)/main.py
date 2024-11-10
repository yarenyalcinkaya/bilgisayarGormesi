import os
from roboflow import Roboflow
from ultralytics import YOLO
import torch

# CUDA ve GPU kullanılabilirliğini kontrol et
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli, imgsz=640)

# Roboflow API anahtarını ve proje detaylarını tanımlayın
rf = Roboflow(api_key="i0TfuVw3529E31X5reQE")
project = rf.workspace("kays-wy3nt").project("kayisi-acaej")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO('runs/detect/train1/weights/yolov8n.pt')  # Eğitilmiş modelin yolunu belirtin

# Test klasörünün ve sonuç klasörünün yolunu belirleyin
test_images_folder = r"C:\Users\ZAY MOTORS\PycharmProjects\kayisiProjesi\kayisi-1\test\images"
results_folder = r"C:\Users\ZAY MOTORS\PycharmProjects\kayisiProjesi\results"
os.makedirs(results_folder, exist_ok=True)

# Modeli eğit
if __name__ == "__main__":
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)

# Test klasöründeki her bir görüntüde tahmin yap ve kaydet
for idx, image_file in enumerate(os.listdir(test_images_folder), start=1):
    image_path = os.path.join(test_images_folder, image_file)

    # Tahminleri yap
    results = model.predict(source=image_path, conf=0.45, iou=0.45)

    # Tahmin edilen görüntüyü kaydet
    results[0].save(os.path.join(results_folder, f"prediction_{idx}.jpg"))
    print(f"Processed {image_file} - Results saved as prediction_{idx}.jpg in 'results' folder")