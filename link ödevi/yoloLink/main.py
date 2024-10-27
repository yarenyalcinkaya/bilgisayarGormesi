import pandas as pd
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
import os
import re
import cv2

# CSV'den linkleri oku
csv_dosya = "linkler.csv"
veriler = pd.read_csv(csv_dosya)


model = YOLO("yolo11n.pt")


tespit_sonuclari = []

# Her bir linkteki fotoğrafı al ve YOLO ile tespit yap
for indeks, satir in veriler.iterrows():
    url = satir['link']  # CSV'deki link sütunu
    print(f"{url} işleniyor...")

    try:

        cevap = requests.get(url)
        sayfa = BeautifulSoup(cevap.content, 'html.parser')

        # Sayfada resim URL'sini bul
        img_etiketi = sayfa.find('img', src=re.compile(r'.*\.jpg'))
        if img_etiketi:
            img_url = img_etiketi['src']


            img_verisi = requests.get(img_url).content
            img_dosya_adi = f"resim_{indeks}.jpg"
            with open(img_dosya_adi, 'wb') as dosya:
                dosya.write(img_verisi)
            print(f"{img_dosya_adi} olarak kaydedildi")


            sonuclar = model([img_dosya_adi])



            for sonuc in sonuclar:
                kutular = sonuc.boxes
                if kutular is None or len(kutular) == 0:
                    print(f"{img_dosya_adi} içinde nesne tespit edilmedi")
                    continue


                for kutu in kutular:
                    sinif = kutu.cls.cpu().numpy()  # tespit edilen sınıf
                    sinif_adi = model.names[int(sinif)]  # sınıf ismi (örneğin insan)

                    print(f"{img_dosya_adi} içinde {sinif_adi} tespit edildi")


                    tespit_sonuclari.append({
                        'url': url,
                        'sinif': sinif_adi
                    })

            # Tespit edilen sonuçları görselleştir ve kaydet
            sonuc_yolu = f"sonuc_{indeks}.jpg"
            sonuc.save(sonuc_yolu)  # sonucu diske kaydet
            print(f"{sonuc_yolu} olarak kaydedildi")

            # OpenCV ile kaydedilen sonucu göster
            resim = cv2.imread(sonuc_yolu)
            if resim is not None:
                cv2.imshow(f"Tespit Sonucu {indeks}", resim)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        else:
            print(f"{url} üzerinde resim bulunamadı")

    except Exception as e:
        print(f"{url} işlenirken hata oluştu: {str(e)}")

print("Tespit işlemi tamamlandı.")