import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8n.pt')  # YOLOv8 model dosyasını yükleyin

# Video dosyasını aç
video_path = 'video.mp4'  # Test etmek istediğiniz video dosyasının yolunu belirtin
cap = cv2.VideoCapture(video_path)

# Sınıf isimlerini tanımlayın
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

while True:
    # Video'dan bir kare oku
    ret, frame = cap.read()

    # Video akışı sona erdiyse döngüyü kır
    if not ret:
        break

    # Nesne tespiti yap
    results = model(frame)

    # Sonuçları görüntüle
    for result in results:
        # Tespit edilen nesneleri çiz
        boxes = result.boxes.xyxy.numpy()  # Sınır kutuları
        confidences = result.boxes.conf.numpy()  # Güven skoru
        class_ids = result.boxes.cls.numpy()  # Sınıf kimlikleri

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)  # Sınır kutusunun koordinatlarını al
            label = f'Class: {class_names[int(class_id)]}, Confidence: {confidence:.2f}'

            # Sınır kutusunu ve etiketi çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sonucu göster
    cv2.imshow('YOLOv8 Object Detection', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video nesnesini serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()