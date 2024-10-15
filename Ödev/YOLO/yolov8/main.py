import cv2
from ultralytics import YOLO

# Model yüklendi
model = YOLO('yolov8n.pt')

# Sınıf isimlerini tanımlandı
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Görüntü okuma
image_path = 'trafik2.jpg'  # Test etmek istediğiniz görüntünün yolu belirtildi
image = cv2.imread(image_path)

# Nesne tespiti yapıldı
results = model(image)

# Sonuçları görüntüleme
for result in results:
    # Tespit edilen nesneleri çiz
    boxes = result.boxes.xyxy.numpy()  # Sınır kutuları
    confidences = result.boxes.conf.numpy()  # Güven skoru
    class_ids = result.boxes.cls.numpy()  # Sınıf kimlikleri

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int)  # Sınır kutusunun koordinatlarını al
        label = f'Class: {class_names[int(class_id)]}, Confidence: {confidence:.2f}'

        # Sınır kutusunu ve etiketi çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Sonucu göster
cv2.imshow('YOLOv8 Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
