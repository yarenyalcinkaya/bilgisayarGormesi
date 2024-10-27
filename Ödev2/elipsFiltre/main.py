import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi yükle
image = cv2.imread('kare.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yatay türev (Sobel)
sobel_horizontal = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Yatay yön
sobel_horizontal_abs = np.abs(sobel_horizontal)

# Dikey türev (Sobel)
sobel_vertical = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)    # Dikey yön
sobel_vertical_abs = np.abs(sobel_vertical)

# Büyüklük hesaplama (magnitude)
magnitude = np.sqrt(np.square(sobel_horizontal) + np.square(sobel_vertical))
magnitude = np.uint8(np.clip(magnitude, 0, 255))  # Değerleri 0-255 arasında sınırla ve uint8'e çevir

# Görselleri yan yana gösterme
plt.figure(figsize=(20, 5))

# Orijinal kare resmi
plt.subplot(1, 4, 1)
plt.title("Orijinal Kare Resmi")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Yatay kenar türevi
plt.subplot(1, 4, 2)
plt.title("Yatay Kenar Türevleri")
plt.imshow(sobel_horizontal_abs, cmap='gray')
plt.axis('off')

# Dikey kenar türevi
plt.subplot(1, 4, 3)
plt.title("Dikey Kenar Türevleri")
plt.imshow(sobel_vertical_abs, cmap='gray')
plt.axis('off')

# Büyüklük görüntüsü
plt.subplot(1, 4, 4)
plt.title("Kenar Büyüklüğü (Magnitude)")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.show()
