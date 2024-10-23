import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi yükle
image = cv2.imread('elips.png')  # Elips resminin dosya yolunu buraya yaz
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yatay türev filtresi [-1, 1]
horizontal_filter = np.array([[-1, 1]])
horizontal_derivative = cv2.filter2D(gray_image, -1, horizontal_filter)

# Dikey türev filtresi [-1] ve [1]
vertical_filter = np.array([[-1], [1]])
vertical_derivative = cv2.filter2D(gray_image, -1, vertical_filter)

# Magnitude (genlik) hesaplama - yatay ve dikey türevlerin birleşimi
magnitude = np.sqrt(np.square(horizontal_derivative) + np.square(vertical_derivative))
magnitude = np.uint8(magnitude)  # Sonuç görüntü formatına çevrilir

# Görselleri yan yana gösterme
plt.figure(figsize=(20, 5))

# Orijinal elips resmi
plt.subplot(1, 4, 1)
plt.title("Orijinal Elips Resmi")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Yatay türev sonucu
plt.subplot(1, 4, 2)
plt.title("Yatay Türev ([-1, 1] filtresi)")
plt.imshow(horizontal_derivative, cmap='gray')
plt.axis('off')

# Dikey türev sonucu
plt.subplot(1, 4, 3)
plt.title("Dikey Türev ([-1], [1] filtresi)")
plt.imshow(vertical_derivative, cmap='gray')
plt.axis('off')

# Yatay ve dikey türevlerin birleşimi (magnitude)
plt.subplot(1, 4, 4)
plt.title("Magnitude (Yatay + Dikey)")
plt.imshow(magnitude, cmap='gray')
plt.axis('off')

plt.show()